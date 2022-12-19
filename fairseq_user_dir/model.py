from dataclasses import dataclass

from icecream import ic

from fairseq.distributed import fsdp_wrap
from fairseq.models.transformer import TransformerConfig
from fairseq.utils import safe_getattr, safe_hasattr
from fairseq.models.transformer_lm import (
    TransformerLanguageModel,
    TransformerLanguageModelConfig,
)
from fairseq.models.transformer.transformer_decoder import (
    TransformerDecoderBase,
    TransformerDecoder,
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
class StaircaseTransformerLanguageModelConfig(TransformerLanguageModelConfig):
    use_alibi: bool = False


@register_model("staircase_lm", dataclass=StaircaseTransformerLanguageModelConfig)
class StaircaseTransformerDecoderModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if safe_getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = safe_getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        if args.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(
                task.source_dictionary,
                eval(args.character_filters),
                args.character_embedding_dim,
                args.decoder_embed_dim,
                args.char_embedder_highway_layers,
            )
        elif args.adaptive_input:
            embed_tokens = AdaptiveInput(
                len(task.source_dictionary),
                task.source_dictionary.pad(),
                args.decoder_input_dim,
                args.adaptive_input_factor,
                args.decoder_embed_dim,
                options.eval_str_list(args.adaptive_input_cutoff, type=int),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            embed_tokens = cls.build_embedding(
                args, task.source_dictionary, args.decoder_input_dim
            )

        if args.tie_adaptive_weights:
            assert args.adaptive_input
            assert args.adaptive_input_factor == args.adaptive_softmax_factor
            assert (
                args.adaptive_softmax_cutoff == args.adaptive_input_cutoff
            ), "{} != {}".format(
                args.adaptive_softmax_cutoff, args.adaptive_input_cutoff
            )
            assert args.decoder_input_dim == args.decoder_output_dim

        use_alibi = args.use_alibi
        cfg = TransformerConfig.from_namespace(args)
        decoder = StaircaseTransformerDecoder(
            cfg,
            task.target_dictionary,
            embed_tokens,
            no_encoder_attn=True,
            use_alibi=use_alibi,
        )
        return cls(decoder)


class StaircaseTransformerDecoder(TransformerDecoderBase):
    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
        use_alibi=False,
    ):
        self.use_alibi = use_alibi
        super().__init__(
            cfg,
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )

    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        if self.use_alibi:
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
