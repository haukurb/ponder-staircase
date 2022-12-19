from dataclasses import dataclass, field
from typing import Optional

from icecream import ic

from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.distributed import fsdp_wrap
from fairseq.models.transformer import TransformerConfig
from fairseq.utils import safe_getattr
from fairseq.models.transformer_lm import (
    TransformerLanguageModel,
    TransformerLanguageModelConfig,
)
from fairseq.modules import AdaptiveInput, CharacterTokenEmbedder
from fairseq.models.transformer.transformer_decoder import (
    TransformerDecoderBase,
)
from fairseq.modules.transformer_layer import TransformerDecoderLayerBase
from fairseq.models import (
    # FairseqLanguageModel,
    register_model,
    register_model_architecture,
)

DEFAULT_MAX_TARGET_POSITIONS = 1024

# TransformerLanguageModel -> fairseq/models/transformer_lm.py
# TransformerDecoder -> fairseq/models/transformer/transformer_decoder.py
# TransformerDecoderBase -> fairseq/models/transformer/transformer_decoder.py
# TransformerDecoderLayerBase -> fairseq/modules/transformer_layer.py
# FairseqLanguageModel -> fairseq/models/fairseq_model.py
# fairseq/model_parallel/models/pipeline_parallel_transformer/model.py

"""
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


to get staircase we just need to subclass this and register it:
- TransformerLanguageModel

to get alibi attention we need to do more surgical things:
- we need to subclass TransformerDecoderBase and overwrite its build_decoder_layer method
- we need to subclass TransformerDecoderLayerBase and overwrite its build_self_attention method 
- we need to subclass TransformerLanguageModel and overwrite its build_model function to use the new TransformerDecoderLayerBase
"""


@dataclass
class StaircaseTransformerConfig(TransformerConfig):
    num_bottom_layers: int = field(default=0)
    num_staircase_layers: int = field(default=4)
    num_top_layers: int = field(default=1)
    use_alibi: bool = field(default=False)

    def __post_init__(self):
        self.decoder.layers = self.num_bottom_layers + self.num_staircase_layers + self.num_top_layers


# @dataclass
# class StaircaseTransformerLanguageModelConfig(TransformerLanguageModelConfig):
#     num_base_layers: int = field(default=0)
#     num_top_layers: int = field(default=1)
#     num_staircase_layers: int = field(default=4)
#     use_alibi: bool = field(default=False)
#     staircase: StaircaseTransformerConfig = field(default_factory=StaircaseTransformerConfig)
#     # pass


# @register_model("staircase_lm", dataclass=StaircaseTransformerLanguageModelConfig)
@register_model("staircase_lm", dataclass=StaircaseTransformerConfig)
class StaircaseTransformerDecoderModel(TransformerLanguageModel):
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
        cfg: TransformerConfig,
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

    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        if cfg.use_alibi:
            assert False
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

    """
    self.forward delegates to this function
    """

    # def extract_features_scriptable(
    #     self,
    #     prev_output_tokens,
    #     encoder_out: Optional[Dict[str, List[Tensor]]],
    #     incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    #     full_context_alignment: bool = False,
    #     alignment_layer: Optional[int] = None,
    #     alignment_heads: Optional[int] = None,
    # ):


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
