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
from typing import Any, Dict, List, Optional

import torch
from fairseq.distributed import fsdp_wrap
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerConfig
from fairseq.models.transformer.transformer_decoder import TransformerDecoderBase
from fairseq.models.transformer_lm import TransformerLanguageModel
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.transformer_layer import TransformerDecoderLayerBase
from fairseq.utils import safe_getattr
from icecream import ic
from torch import Tensor


@dataclass
class StaircaseCache:
    finalized_chunks: List[torch.Tensor] = field(default_factory=list)

    @property
    def num_vecs(self):
        cache_len = sum(chunk.shape[0] for chunk in self.finalized_chunks)
        return cache_len


@dataclass
class StaircaseTransformerConfig(TransformerConfig):
    num_bottom_layers: int = field(default=0)
    num_staircase_layers: int = field(default=4)
    num_top_layers: int = field(default=1)
    use_alibi: bool = field(default=False)

    def __post_init__(self):
        self.decoder.layers = (
            self.num_bottom_layers + self.num_staircase_layers + self.num_top_layers
        )


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

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        assert cfg.decoder.input_dim == cfg.decoder.embed_dim
        embed_tokens = cls.build_embedding(
            cfg, task.source_dictionary, cfg.decoder.embed_dim
        )

        # cfg = StaircaseTransformerConfig.from_namespace(args)
        decoder = StaircaseTransformerDecoder(
            cfg,
            task.target_dictionary,
            embed_tokens,
            no_encoder_attn=True,
        )
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

        end modification of original function
        """

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # Prevent torchscript exporting issue for dynamic quant embedding
        prev_output_tokens = prev_output_tokens.contiguous()
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

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

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        # for idx, layer in enumerate(self.layers):
        for idx, layer in enumerate(self.bottom_layers):
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

        _, _, edim = x.shape

        # we need to make sure that attention for incremental_state behaves the same way as staircase attention
        assert incremental_state is None, "Not implemented yet"
        _x_before_staircase = x
        # used a fixed chunk size as scaffolding for now
        fixed_chunk_len = 4
        # x is of shape [SeqLen, Batch, NumHidden]
        chunks = x.split(fixed_chunk_len, dim=0)
        chunk_lens = [chunk.shape[0] for chunk in chunks]

        cache = StaircaseCache()

        # queue chunked input so that we only input one chunk a at a time
        chunk_queue = deque(chunks)
        total_staircase_forwards = len(chunks) + len(self.staircase_layers) - 1

        # TODO: only roll staircase while training
        staircase_start = torch.randint(0, len(self.staircase_layers), size=[1])[0]
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
        ncached_chunks = (
            0  # number of chunks removed from staircase after they are finalized
        )
        for _forward_count, layer in enumerate(
            islice(cycle(rolled_staircase), total_staircase_forwards), start=1
        ):
            if chunk_queue:
                next_chunk = chunk_queue.popleft()
                x = torch.cat([x, next_chunk], dim=0)
                nchunks_in_staircase += 1

            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # XXX TODO: self_attn_mask will not work during inference (we don't properly handle incremental_state)
            assert incremental_state is None
            nqueries = x.shape[0]
            curr_self_attn_mask = self_attn_mask
            if cache.num_vecs > 0:
                assert isinstance(self_attn_mask, torch.FloatTensor)
                # mask is additive, not multiplicative
                curr_self_attn_mask = torch.cat([self_attn_mask.new_zeros(nqueries, cache.num_vecs), self_attn_mask], dim=1)

            # self_attn_padding_mask: [Batch, KeySeqLen]
            curr_self_attn_padding_mask = self_attn_padding_mask
            if self_attn_padding_mask is not None:
                curr_self_attn_padding_mask = self_attn_padding_mask[:, :cache.num_vecs + nqueries]

            x, layer_attn, _ = layer(
                x,
                None,  # enc_out
                None,  # encoder_padding_mask
                incremental_state,
                self_attn_mask=curr_self_attn_mask,
                self_attn_padding_mask=curr_self_attn_padding_mask,
                need_attn=False,
                need_head_weights=False,
                key_cache=cache,
            )
            # XXX: if we skip a layer due to layerdrop, we still want to count it in staircase_nforwards,
            #      that means we cannot use layerdropmodule from fairseq to implement our layerdrop

            # after a chunk as been forwarded num_staircase_layers many times it becomes a key_only chunk
            # we find the first chunk where it hasnt been forwarded that many times
            nforwards_per_chunk[ncached_chunks:nchunks_in_staircase] += 1
            if nforwards_per_chunk.ge(len(self.staircase_layers)).any():
                # check if any chunk as been forwarded enough times to leave the staircase
                nchunks_finished_this_forward = (
                    nforwards_per_chunk[ncached_chunks:]
                    .ge(len(self.staircase_layers))
                    .sum()
                )
                # cache all the vectors from those chunks (can be mnore than one chunk)
                nvecs_to_be_cached = sum(
                    chunk_lens[
                        ncached_chunks : ncached_chunks + nchunks_finished_this_forward
                    ]
                )
                # partition x into next inputs and cached inputs
                cache.finalized_chunks.append(x[:nvecs_to_be_cached])
                x = x[nvecs_to_be_cached:]
                # keep track of all chunks forwarded
                ncached_chunks += nchunks_finished_this_forward

        assert x.shape[0] == 0, "All have been moved to the cache at this point"
        x = torch.cat(cache.finalized_chunks, dim=0)

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
        key_cache: Optional[StaircaseCache] = None,
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
        # modification of original function
        assert (
            not self.cross_self_attention
        ), "self.cross_self_attention does not behave correctly for rotary embeddings"
        # end modification
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
        if key_cache is None:
            x_with_cache = x
        else:
            x_with_cache = torch.cat(key_cache.finalized_chunks + [x], dim=0)
        # XXX TODO: add attn_mask and key_padding_mask back in
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