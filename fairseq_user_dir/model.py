"""
This file is borrows heavily from fairseq, specifically from the following files:
   - fairseq/models/transformer_lm.py
   - fairseq/models/transformer/transformer_decoder.py
   - fairseq/modules/transformer_layer.py
   - fairseq/models/fairseq_model.py

which has the following license:
    MIT License

    Copyright (c) Facebook, Inc. and its affiliates.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""
from collections import deque
from dataclasses import dataclass, field
from itertools import cycle, islice
from typing import Any, Dict, List, Optional, Union

import torch
from fairseq.distributed import fsdp_wrap
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerConfig
from fairseq.models.transformer.transformer_decoder import TransformerDecoderBase
from fairseq.modules import MultiheadAttention
from fairseq.models.transformer_lm import TransformerLanguageModel
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.transformer_layer import TransformerDecoderLayerBase
from fairseq.dataclass import ChoiceEnum
from fairseq.utils import safe_getattr, set_torch_seed
from icecream import ic
from torch import Tensor

from omegaconf import II

import numpy as np

from . import multihead_rotary_attention
from .piecewise_schedule import PiecewiseBooleanFn, PiecewiseLinearFn

# import

"""
Our alternatives for (length extrapolatable) rotary encodings are:
https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
https://github.com/sunyt32/torchscale/blob/main/torchscale/component/multihead_attention.py
https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/layers/rotary.py
"""
POSITION_ENCODING_CHOICES = ChoiceEnum(
    [
        "xpos",
        "rope",
        "none",
    ]
)


# XXX: we should add optional forgetting
class StaircaseCache:
    """XXX: If max_context_size is short (under 128) it causes nans in the outputs of the staircase layers for chomsky tasks.
    We assume this is because of activation norm magnification and exploding gradients and the fact that most of the loss
    is masked (except a tiny bit at the end).  This does not seem to be a problem for bookcorpusopen models for the
    tested input size, context size, chunk size and model height.

    Aside from differing layer normalization strategies (decouple layernorm from layers for example).
    We could be solve this by applying a context size curriculum:
    - schedule provided as piecewise linear function or similar
    - purely dynamic; if no nan is found for k steps, narrow the context size (until target context size is reach)
    """

    def __init__(self, *, max_context_size: Optional[int] = None):
        self.finalized_chunks: List[torch.Tensor] = []
        self.unmerged_chunks: List[torch.Tensor] = []
        self._depths: List[int] = []
        self.chunk_lengths: List[int] = []
        self.max_context_size = max_context_size if max_context_size >= 0 else None
        self.max_unconsolidated: int = 4
        self.valid_cache = True  # for cache invalidation
        # we don't need to recompute state tensor if nothing has been added to the cache
        self.cached_state = None

    @property
    def num_forgotten(self):
        state = self.get_state()
        if state is None:
            return 0
        return self.length - state.shape[0]

    @property
    def length(self):
        cache_len = sum(chunk.shape[0] for chunk in self.unmerged_chunks)
        return cache_len

    def consolidate(self):
        """Merges chunks on the right side of the cache when enough have been added.
        We want to minimize the depth of each chunk in the merge tree for faster backprop time, so we track chunk depths
        and merge accordingly.
        Effectiveness with max_unconsolidated in (2,4,8):
            - A 3-8-1 transformer (dim=640) on seq_len=100 and bsz=100 we get 20% higher updates per second with chunk_size=1 (1.22 -> 1.50).
            - with chunk_size=2 it is still between ~18% and 19% speedup.
        """
        consol_start = self._depths.index(self._depths[-1]) if self._depths else 0
        length_unconsolidated = len(self._depths) - consol_start
        if length_unconsolidated < self.max_unconsolidated:
            return
        self.finalized_chunks[consol_start:] = [
            torch.cat(self.finalized_chunks[consol_start:])
        ]
        # consolidate last chunk and increment its depth
        self._depths[consol_start:] = [self._depths[-1] + 1]
        # recursively consolidate if we can
        self.consolidate()

    def get_state(self):
        if self.valid_cache:
            return self.cached_state

        if self.max_context_size is None:
            return torch.cat(self.finalized_chunks, dim=0)
        elif self.max_context_size == 0:
            return None
        # this could be optimized with consolidation (for up to 10-15% speedup for very long contexts~128)
        if self.unmerged_chunks:
            num_chunks_from_tail = (
                np.searchsorted(
                    # reverse list, find first index of cumulative length that exceeds context size
                    np.cumsum(self.chunk_lengths[-self.max_context_size :][::-1]),
                    self.max_context_size,
                )
                + 1
            )
            # we add one because cumsum isn't prepended with zero
            return torch.cat(self.unmerged_chunks[-num_chunks_from_tail:], dim=0)
        return None

    def add(self, chunk):
        self.finalized_chunks.append(chunk)
        self.unmerged_chunks.append(chunk)
        self.chunk_lengths.append(chunk.shape[0])
        self._depths.append(0)
        self.consolidate()
        self.valid_cache = False
        _ = self.get_state()

    def get_global_state(self):
        return torch.cat(self.unmerged_chunks, dim=0)


@dataclass
class StaircaseTransformerConfig(TransformerConfig):
    num_bottom_layers: int = field(default=0)
    num_staircase_layers: int = field(default=4)
    num_top_layers: int = field(default=1)
    use_alibi: bool = field(default=False)
    use_fixed_chunking: Optional[str] = field(default="0")
    chunk_length_parameter: Optional[str] = field(default="0")
    valid_use_fixed_chunking: Optional[str] = II("model.use_fixed_chunking")
    valid_chunk_length_parameter: Optional[str] = II("model.chunk_length_parameter")

    position_encoding: POSITION_ENCODING_CHOICES = field(  # type: ignore
        default="xpos",
    )
    max_context_size: Optional[str] = field(default="4")
    roll_staircase: bool = field(default=True)
    recurrent_stride: Optional[str] = field(default="4")
    # proportion of samples that will execute in normal mode
    standard_prob: Optional[float] = field(default=None)

    def __post_init__(self):
        self.decoder.layers = (
            self.num_bottom_layers + self.num_staircase_layers + self.num_top_layers
        )
        self.decoder.layers = (
            self.num_bottom_layers + self.num_staircase_layers + self.num_top_layers
        )
        # so we can build the model without going through hydra
        if self.valid_use_fixed_chunking == II("model.valid_use_fixed_chunking"):
            self.valid_use_fixed_chunking = self.use_fixed_chunking
        if self.valid_chunk_length_parameter == II(
            "model.valid_chunk_length_parameter"
        ):
            self.valid_chunk_length_parameter = self.chunk_length_parameter


@register_model("staircase_lm", dataclass=StaircaseTransformerConfig)
class StaircaseTransformerDecoderModel(TransformerLanguageModel):
    """structure of arch=transformer_lm:
    TransformerLanguageModel registers the arch "transformer_lm"
    TransformerLanguageModel inherits from FairseqLanguageModel
    TransformerLanguageModel builds TransformerDecoder
        TransformerDecoder inherits from TransformerDecoderBase which inherits from FairseqIncrementalDecoder
            # TransformerDecoder is just a thin wrapper
            TransformerDecoderBase
                # calls build_decoder_layer and delegates to transformer_layer.TransformerDecoderLayerBase
                # has the actual decoder forward pass
                TransformerDecoderLayerBase
                    # builds the multiheadattention

    to get staircase we just need to subclass the following and register it:
    - TransformerLanguageModel

    to get alibi attention we need to do more surgical things:
    - we need to subclass TransformerDecoderBase and overwrite its build_decoder_layer method
    - we need to subclass TransformerDecoderLayerBase and overwrite its build_self_attention method
    - we need to subclass TransformerLanguageModel and overwrite its build_model function to use the new TransformerDecoderLayerBase

    where the modules are located in:
    - TransformerLanguageModel -> fairseq/models/transformer_lm.py
    - TransformerDecoder -> fairseq/models/transformer/transformer_decoder.py
    - TransformerDecoderBase -> fairseq/models/transformer/transformer_decoder.py
    - TransformerDecoderLayerBase -> fairseq/modules/transformer_layer.py
    - FairseqLanguageModel -> fairseq/models/fairseq_model.py
    - fairseq/model_parallel/models/pipeline_parallel_transformer/model.py
    """

    def __init__(self, decoder):
        super().__init__(decoder)
        self.num_updates = 0

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        assert cfg.decoder.input_dim == cfg.decoder.embed_dim
        embed_tokens = cls.build_embedding(
            cfg, task.source_dictionary, cfg.decoder.embed_dim
        )

        cfg.decoder.layers = (
            cfg.num_bottom_layers + cfg.num_staircase_layers + cfg.num_top_layers
        )
        decoder = StaircaseTransformerDecoder(
            cfg,
            task.source_dictionary,
            embed_tokens,
            no_encoder_attn=True,
        )
        # we use this to have reproducible RNG across multiple GPUs that needs to be the same
        # on a given time step
        return cls(decoder)


class StaircaseTransformerDecoder(TransformerDecoderBase):
    def __init__(
        self,
        cfg: StaircaseTransformerConfig,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(
            cfg,
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )
        assert self.cfg.decoder.layerdrop == 0, "Not implemented yet"
        bottom_range = list(range(cfg.num_bottom_layers))
        staircase_range = list(
            range(len(bottom_range), len(bottom_range) + cfg.num_staircase_layers)
        )
        top_range = range(
            len(bottom_range + staircase_range),
            len(bottom_range + staircase_range) + cfg.num_top_layers,
        )
        self.bottom_layers, self.staircase_layers, self.top_layers = [], [], []
        for i in bottom_range:
            self.bottom_layers.append(self.layers[i])
        for i in staircase_range:
            self.staircase_layers.append(self.layers[i])
        for i in top_range:
            self.top_layers.append(self.layers[i])

        self.curriculum_use_fixed_chunking = PiecewiseBooleanFn.from_string(
            self.cfg.use_fixed_chunking
        )
        self.curriculum_chunk_length_parameter = PiecewiseLinearFn.from_string(
            self.cfg.chunk_length_parameter
        )
        self.curriculum_recurrent_stride = PiecewiseLinearFn.from_string(
            self.cfg.recurrent_stride
        )
        self.curriculum_context_size = PiecewiseLinearFn.from_string(
            self.cfg.max_context_size
        )
        self.curriculum_valid_use_fixed_chunking = PiecewiseBooleanFn.from_string(
            self.cfg.valid_use_fixed_chunking
        )
        self.curriculum_valid_chunk_length_parameter = PiecewiseLinearFn.from_string(
            self.cfg.valid_chunk_length_parameter
        )

        self.use_fixed_chunking = None
        self.recurrent_stride = None
        self.max_context_size = None
        self.chunk_length_parameter = None
        self.valid_use_fixed_chunking = None
        self.valid_chunk_length_parameter = None

    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        if cfg.use_alibi:
            assert False, "Not implemented yet"
            layer = TransformerAlibiDecoderLayerBase(cfg, no_encoder_attn)
        else:
            layer = StaircaseTransformerDecoderLayerBase(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)  # type: ignore
        else:
            return features

    """
    self.forward delegates to this function

    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bsz, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        encoder_padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            encoder_padding_mask = encoder_out["encoder_padding_mask"][0]

        """
        begin modification of original function:

        XXX: consider adding cyclic absolute positional embeddings to test hypothesis of chomsky hierarchy paper
        also consider making those cyclic absolute positional embeddings disjoint i.e. it is not limited to:
            0 1 2 3 4 5 6 7 8...
        but can also be:
            0 1 2 3 7 8 9 10 11..
                    ^--- 4,5,6 were skipped

        """

        # # embed positions
        # positions = None
        # if self.embed_positions is not None:
        #     positions = self.embed_positions(
        #         prev_output_tokens, incremental_state=incremental_state
        #     )

        # if incremental_state is not None:
        #     prev_output_tokens = prev_output_tokens[:, -1:]
        #     if positions is not None:
        #         positions = positions[:, -1:]

        """
        end modification of original function
        """

        # Prevent torchscript exporting issue for dynamic quant embedding
        prev_output_tokens = prev_output_tokens.contiguous()
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        # if positions is not None:
        #     x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        """
        begin modification of original function:
        """

        bottom_layers = list(self.bottom_layers)
        use_staircase = bool(self.staircase_layers)
        with set_torch_seed(self.num_updates):
            if (
                self.training
                and (self.cfg.standard_prob is not None)
                and (np.random.rand() < self.cfg.standard_prob)
            ):
                use_staircase = False
                bottom_layers.extend(self.staircase_layers)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(bottom_layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None
            x, layer_attn, _ = layer(
                x,
                None,  # enc_out
                None,  # encoder_padding_mask
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=False,
                need_head_weights=False,
            )
            inner_states.append(x)

        # start of staircase part of decoder
        if use_staircase:
            _, _, edim = x.shape

            # we need to make sure that attention for incremental_state behaves the same way as staircase attention
            assert incremental_state is None, "Not implemented yet"
            _x_before_staircase = x
            # used a fixed chunk size as scaffolding for now
            fixed_chunk_len = 4

            use_fixed_chunking = (
                self.use_fixed_chunking
                if self.training
                else self.valid_use_fixed_chunking
            )
            split_param = (
                self.chunk_length_parameter
                if self.training
                else self.valid_chunk_length_parameter
            )
            split_param = int(split_param) if use_fixed_chunking else split_param
            if not use_fixed_chunking:
                rng = np.random.default_rng(seed=hash(prev_output_tokens) % 2**31)
                chunk_lens = rng.poisson(lam=split_param, size=slen)
                cumulative = np.cumsum(chunk_lens)
                cutoff = cumulative.searchsorted(slen)
                last_chunk_len = slen - cumulative[cutoff - 1]
                chunk_lens = chunk_lens[:cutoff].tolist()  # type: ignore
                chunk_lens.append(last_chunk_len)
                split_param = chunk_lens

            # x is of shape [SeqLen, Batch, NumHidden]
            chunks = x.split(split_param, dim=0)
            chunk_lens = [chunk.shape[0] for chunk in chunks]

            # to store vectors when they reach the output side of the staircase
            cache = StaircaseCache(max_context_size=self.max_context_size)
            # cache = StaircaseCache(max_context_size=self.cfg.max_context_size)

            # queue chunked input so that we only input one chunk a at a time
            chunk_queue = deque(chunks)
            total_staircase_forwards = len(chunks) + len(self.staircase_layers) - 1

            # roll staircase randomly during training, this might not be needed
            staircase_start = 0
            if self.training and self.cfg.roll_staircase:
                staircase_start = torch.randint(
                    0, len(self.staircase_layers), size=[1]
                )[0]
            rolled_staircase = (
                self.staircase_layers[staircase_start:]
                + self.staircase_layers[:staircase_start]
            )

            # track the number of forwards for each chunk so we know when they are finalized
            nforwards_per_chunk = torch.zeros(len(chunks), dtype=torch.long)
            x = x.new_zeros(0, bsz, edim)  # here x is equivalent to 'prev_layer_output'
            nchunks_in_staircase = (
                0  # number of chunks removed from the queue and put into the staircase
            )
            nfinished_chunks = (
                0  # number of chunks removed from staircase after they are finalized
            )

            # the maximum number of forwards we need to make
            myiter = islice(
                cycle(rolled_staircase),
                len(chunks) * max(len(self.staircase_layers), self.recurrent_stride),
            )
            # myiter = islice(cycle(rolled_staircase), len(chunks) * max(len(self.staircase_layers), self.cfg.recurrent_stride))

            recurrence_counter = self.recurrent_stride
            # recurrence_counter = self.cfg.recurrent_stride
            while True:
                # ic(nforwards_per_chunk[:nchunks_in_staircase+1])
                layer = next(myiter)

                if chunk_queue and self.recurrent_stride <= recurrence_counter:
                    # if chunk_queue and self.cfg.recurrent_stride <= recurrence_counter:
                    recurrence_counter = 0
                    next_chunk = chunk_queue.popleft()
                    x = torch.cat([x, next_chunk], dim=0)
                    nchunks_in_staircase += 1

                if incremental_state is None and not full_context_alignment:
                    self_attn_mask = self.buffered_future_mask(x)
                else:
                    self_attn_mask = None

                assert self_attn_mask is not None
                # XXX TODO: self_attn_mask will not work during inference (we don't properly handle incremental_state)
                assert incremental_state is None
                extra_state = cache.get_state()
                nqueries = x.shape[0]
                curr_self_attn_mask = self_attn_mask
                if extra_state is not None:
                    assert self_attn_mask.dtype in (
                        torch.float16,
                        torch.float32,
                        torch.float64,
                    )
                    # assert isinstance(self_attn_mask, torch.FloatTensor) or isinstance(self_attn_mask, torch.FloatTensor)
                    # mask is additive, not multiplicative, so a value of zero means a given position is included
                    # shape: [Time, Batch, Embed]
                    num_extra_keys = extra_state.shape[0]
                    # mask shape: [NumQueries, NumKeys]
                    curr_self_attn_mask = torch.cat(
                        [
                            self_attn_mask.new_zeros(nqueries, num_extra_keys),
                            self_attn_mask,
                        ],
                        dim=1,
                    )

                curr_self_attn_padding_mask = self_attn_padding_mask
                if self_attn_padding_mask is not None:
                    # mask shape: [Batch, Time]
                    curr_self_attn_padding_mask = self_attn_padding_mask[
                        :, cache.num_forgotten : cache.length + nqueries
                    ]

                x, layer_attn, _ = layer(
                    x,
                    None,  # enc_out
                    None,  # encoder_padding_mask
                    incremental_state,
                    self_attn_mask=curr_self_attn_mask,
                    self_attn_padding_mask=curr_self_attn_padding_mask,
                    need_attn=False,
                    need_head_weights=False,
                    extra_state=extra_state,
                )

                if x.isnan().any() and self_attn_padding_mask is not None:
                    # xformer attentions have a nan bug when using masks: https://github.com/facebookresearch/xformers/issues/631
                    # [Batch, Time] -> [Time, Batch]; to match x
                    query_padding_mask = self_attn_padding_mask[
                        :, cache.length : cache.length + nqueries
                    ].transpose(0, 1)
                    x[query_padding_mask] = 0

                # XXX: if we skip a layer due to layerdrop, we still want to count it in staircase_nforwards,
                #      that means we cannot use layerdropmodule from fairseq to implement our layerdrop

                # After a chunk as been forwarded num_staircase_layers many times it is finalized and becomes
                # a key_only chunk.  We find the first chunk where it hasnt been forwarded that many times.
                nforwards_per_chunk[nfinished_chunks:nchunks_in_staircase] += 1
                if nforwards_per_chunk.ge(len(self.staircase_layers)).any():
                    # check if any chunk as been forwarded enough times to leave the staircase
                    nchunks_finished_this_forward = (
                        nforwards_per_chunk[nfinished_chunks:]
                        .ge(len(self.staircase_layers))
                        .sum()
                    )
                    # cache all the vectors from those chunks (can be mnore than one chunk)
                    nvecs_to_be_cached = sum(
                        chunk_lens[
                            nfinished_chunks : nfinished_chunks
                            + nchunks_finished_this_forward
                        ]
                    )
                    # partition x into next inputs and cached inputs
                    cache.add(x[:nvecs_to_be_cached])
                    x = x[nvecs_to_be_cached:]
                    # keep track of all chunks forwarded
                    nfinished_chunks += nchunks_finished_this_forward
                if x.shape[0] == 0 and not chunk_queue:
                    break
                recurrence_counter += 1

            assert x.shape[0] == 0, "All have been moved to the cache at this point"
            x = cache.get_global_state()
            inner_states.append(x)
        # end of staircase part of decoder

        for idx, layer in enumerate(self.top_layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None
            x, layer_attn, _ = layer(
                x,
                None,  # enc_out
                None,  # encoder_padding_mask
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=False,
                need_head_weights=False,
            )
            inner_states.append(x)

        """
        end modification of original function
        """

        if attn is not None:
            # average probabilities over heads
            attn = attn.mean(dim=0)  # type:ignore

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def set_num_updates(self, num_updates):
        # copy from fairseq.models.fairseq_model
        """State from trainer to pass along to model at every update."""
        self.num_updates = num_updates
        self.use_fixed_chunking = self.curriculum_use_fixed_chunking(num_updates)
        self.recurrent_stride = self.curriculum_recurrent_stride(num_updates)
        self.chunk_length_parameter = self.curriculum_chunk_length_parameter(
            num_updates
        )
        self.valid_use_fixed_chunking = self.curriculum_valid_use_fixed_chunking(
            num_updates
        )
        self.valid_chunk_length_parameter = (
            self.curriculum_valid_chunk_length_parameter(num_updates)
        )
        self.max_context_size = self.curriculum_context_size(num_updates)

        for m in self.modules():
            if hasattr(m, "set_num_updates") and m != self:
                m.set_num_updates(num_updates)  # type: ignore


class StaircaseTransformerDecoderLayerBase(TransformerDecoderLayerBase):
    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        extra_state: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)

        """
        begin modification of original function
        """

        assert (
            not self.cross_self_attention
        ), "self.cross_self_attention does not behave correctly for rotary embeddings"

        """
        end modification
        """

        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        """
        begin modification of original function
        """
        if extra_state is None:
            x_with_cache = x
        else:
            x_with_cache = torch.cat([extra_state, x], dim=0)
        x, attn = self.self_attn(
            query=x,
            key=x_with_cache,
            value=x_with_cache,
            incremental_state=incremental_state,
            need_weights=False,
            key_padding_mask=self_attn_padding_mask,
            attn_mask=self_attn_mask,
        )
        """
        end modification of original function
        """

        if self.c_attn is not None:
            tgt_len, bsz = x.size(0), x.size(1)
            x = x.view(tgt_len, bsz, self.nh, self.head_dim)
            x = torch.einsum("tbhd,h->tbhd", x, self.c_attn)
            x = x.reshape(tgt_len, bsz, self.embed_dim)
        if self.attn_ln is not None:
            x = self.attn_ln(x)
        x = self.dropout_module(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)  # type: ignore
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)  # type: ignore

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def build_self_attention(
        self, embed_dim, cfg, add_bias_kv=False, add_zero_attn=False
    ):
        return multihead_rotary_attention.MultiheadRotaryAttention(
            # return MultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            dropout=cfg.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            # when true this ignores the k and v vectors and uses q instead
            self_attention=False,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            xformers_att_config=cfg.decoder.xformers_att_config,
            # our addition
            position_encoding=cfg.position_encoding,
        )


class TransformerAlibiDecoderLayerBase(TransformerDecoderLayerBase):
    # def build_self_attention(
    #     self, embed_dim, cfg, add_bias_kv=False, add_zero_attn=False
    # ):
    #     return MultiheadAttention(
    #         embed_dim,
    #         cfg.decoder.attention_heads,
    #         kdim=cfg.encoder.embed_dim,
    #         vdim=cfg.encoder.embed_dim,
    #         dropout=cfg.attention_dropout,
    #         encoder_decoder_attention=True,
    #         q_noise=self.quant_noise,
    #         qn_block_size=self.quant_noise_block_size,
    #         xformers_att_config=cfg.encoder.xformers_att_config,
    #     )
    pass
