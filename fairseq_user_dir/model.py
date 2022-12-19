from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import deque
from itertools import cycle, islice

from torch import Tensor
import torch

from icecream import ic

from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.distributed import fsdp_wrap
from fairseq.models.transformer import TransformerConfig
from fairseq.utils import safe_getattr
from fairseq.models.transformer_lm import (
    TransformerLanguageModel,
)
from fairseq.models.transformer.transformer_decoder import (
    TransformerDecoderBase,
)
from fairseq.modules.transformer_layer import TransformerDecoderLayerBase
from fairseq.models import (
    register_model,
    register_model_architecture,
)


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
    """ structure of arch=transformer_lm:
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
            layer = TransformerDecoderLayerBase(cfg, no_encoder_attn)
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

        _x_before_staircase = x
        # used a fixed chunk size as scaffolding for now
        fixed_chunk_len = 4
        chunks = x.split(
            fixed_chunk_len, dim=0
        )  # x is of shape [SeqLen, Batch, NumHidden]

        chunk_queue = deque(chunks)
        num_forwards = len(self.staircase_layers) + len(chunks)
        staircase_start = torch.randint(0, len(self.staircase_layers), size=[1])[0]
        rolled_staircase = (
            self.staircase_layers[staircase_start:]
            + self.staircase_layers[:staircase_start]
        )
        x = x.new_zeros(0, bsz, edim)  # here x is equivalent to 'prev_layer_output'
        for _nforwards, layer in enumerate(
            islice(cycle(rolled_staircase), num_forwards)
        ):
            if chunk_queue:
                x = torch.cat([x, chunk_queue.popleft()], dim=0)

            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            curr_slen = x.shape[0]
            curr_self_attn_padding_mask = (
                self_attn_padding_mask[:, :curr_slen]
                if self_attn_padding_mask is not None
                else None
            )

            x, layer_attn, _ = layer(
                x,
                None,  # enc_out
                None,  # encoder_padding_mask
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=curr_self_attn_padding_mask,
                need_attn=False,
                need_head_weights=False,
            )
            inner_states.append(x)

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


# from fairseq.modules import LayerNorm, MultiheadAttention
# @with_incremental_state
# class MultiheadAttention(nn.Module):
#     """Multi-headed attention.
