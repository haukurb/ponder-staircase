"""
This file is borrows heavily from fairseq, specifically from the following file:
   - from fairseq.modules.multihead_attention import MultiheadAttention

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

from typing import Optional, Tuple
import logging

import torch
from torch import Tensor

from fairseq.modules.multihead_attention import MultiheadAttention, _mask_for_xformers
from fairseq.incremental_decoding_utils import with_incremental_state
from xformers.components.attention import build_attention
from xformers.components.attention.utils import maybe_merge_masks

from icecream import ic

from . import torchscale_xpos_relative_position

logger = logging.getLogger(__file__)

# @with_incremental_state
class MultiheadRotaryAttention(MultiheadAttention):

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        xformers_att_config: Optional[str] = None,
        xformers_blocksparse_layout: Optional[
            torch.Tensor
        ] = None,
        xformers_blocksparse_blocksize: Optional[
            int
        ] = 16,
        position_encoding="xpos",
    ):
        super().__init__(
            embed_dim,
            num_heads,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=self_attention,
            encoder_decoder_attention=encoder_decoder_attention,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            xformers_att_config=xformers_att_config,
            xformers_blocksparse_layout = xformers_blocksparse_layout,
            xformers_blocksparse_blocksize= 16,
        )
        assert kdim is None
        assert vdim is None
        self.rotary_embedding = None
        if position_encoding in ("xpos", "rope"):
            use_scaling = position_encoding=="xpos"
            self.rotary_embedding = torchscale_xpos_relative_position.XPos(head_dim=embed_dim, use_scaling=position_encoding=="xpos")

    def _xformers_attn_forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        tgt_len, bsz, embed_dim = query.size()
        """
        begin modification of original function
        """
        # if key_padding_mask is not None:
        #     assert key_padding_mask.size(0) == bsz
        #     assert key_padding_mask.size(1) == tgt_len
        """
        end modification
        """

        if self.self_attention:
            key = query
            value = query
        elif self.encoder_decoder_attention:
            value = key

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        """
        begin modification of original function
        """
        if self.rotary_embedding is not None:
            q_len = q.shape[0]
            k_len = k.shape[0]
            # we shift query vectors so that the last query vector has the same encoded position as the last key vector
            offset = (k_len - q_len)//2 + 1
            q = self.rotary_embedding.apply(q, offset = offset)
            k = self.rotary_embedding.apply(k, invert_scale=True)
        """
        end modification
        """

        if self.bias_k is not None:
            assert self.bias_v is not None
            k, v, attn_mask, key_padding_mask = self._add_bias(
                k, v, attn_mask, key_padding_mask, bsz
            )

        def fold_heads(x):
            return (
                x.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        def split_heads(x):
            return (
                x.contiguous()
                .view(-1, bsz, self.num_heads, self.head_dim)
                .transpose(0, 1)
                .transpose(1, 2)
            )

        massage = split_heads if self.attention.requires_head_dimension else fold_heads
        q = massage(q)
        if k is not None:
            k = massage(k)
        if v is not None:
            v = massage(v)

        if self.add_zero_attn:
            k, v, key_padding_mask, attn_mask = self._append_zero_attn(
                k=k, v=v, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )

        kwargs = {}

        if attn_mask is not None and self.attention.supports_attention_mask:
            attn_mask = _mask_for_xformers(attn_mask, to_dtype=q.dtype)
            kwargs["att_mask"] = attn_mask

        if key_padding_mask is not None:
            key_padding_mask = _mask_for_xformers(key_padding_mask, to_dtype=torch.bool)
            if not self.attention.requires_separate_masks:
                attn_mask = maybe_merge_masks(
                    attn_mask,
                    key_padding_mask,
                    batch_size=bsz,
                    src_len=k.size(-2),
                    tgt_len=q.size(-2),
                    num_heads=self.num_heads,
                )
                key_padding_mask = None
                kwargs["att_mask"] = attn_mask
            if self.attention.supports_key_padding_mask:
                kwargs["key_padding_mask"] = key_padding_mask

        y = self.attention(q, k, v, **kwargs)

        y = (
            y.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .flatten(start_dim=2, end_dim=3)
            .transpose(0, 1)
        )
        assert list(y.size()) == [tgt_len, bsz, embed_dim]

        # Dropout not needed because already applied in attention.
        # It is applied to the attention weights before matmul with v.
        y = self.out_proj(y)

        # TODO: support returning attention weights if needed.
        return y, None