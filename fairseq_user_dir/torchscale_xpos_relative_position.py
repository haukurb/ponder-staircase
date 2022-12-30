"""
This file is heavily based on a file from this repository:
  https://github.com/sunyt32/torchscale
Which has the following license

   MIT License

   Copyright (c) Microsoft Corporation.

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
   SOFTWARE
"""

import torch
import torch.nn as nn

from icecream import ic


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def fixed_pos_embedding(x):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq).to(x)
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


class XPos(nn.Module):
    def __init__(
        self, head_dim, scale_base = 512, use_scaling=True,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )
        self.use_scaling = use_scaling

    def forward(self, len, offset=0):
        min = -len // 2
        max = len + min
        scale = self.scale ** torch.arange(min + offset, max + offset, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)
        return (sin, cos, scale)

    def apply(self, x, offset=0, dim=-2, invert_scale=False):
        """
        Note: we might want to implement partial rotation (e.g. rotation only half the dimensions of x instead of all)
        as shown by:
            https://wandb.ai/eleutherai/neox/reports/Partial-Rotary-Tests-v2--Vmlldzo2MjE4MTQ
            https://wandb.ai/lucidrains/x-transformers-experiments/reports/partial-rotary--Vmlldzo2MjAyODY?accessToken=f657029yibaln2vseli6egxxwykhpeuedeuvcnmmdgn4i6d5b1r30it3mp0gw0k5
        """
        bsz, seqlen, _ = x.shape
        sin, cos, scale = self.forward(seqlen)
        # [SeqLen, NumDims/2] -> [1, SeqLen, NumDims]
        sin = sin.repeat(1, 2).unsqueeze(0)
        cos = cos.repeat(1, 2).unsqueeze(0)
        scale = scale.repeat(1, 2).unsqueeze(0)
        xcos = cos * x
        xsin = sin * rotate_half(x)
        if not self.use_scaling:
            return xcos + xsin
        if invert_scale:
            return (xcos + xsin) * (scale ** -1)
        return (xcos + xsin) * scale


def main():
    # just for debugging
    from icecream import ic
    B, T1, T2, C = 1, 1, 30, 6
    x = torch.ones(B, T1, C).float()
    y = torch.ones(B, T2, C).float()
    rotemb = XPos(head_dim=C)
    sin, cos, scale = rotemb(T1)
    # ic(sin, cos, scale)

    longer = 10
    shorter = 3
    sin, cos, scale = rotemb(longer, offset=0)
    longer_scale = scale
    ic(scale[:,0])
    sin, cos, scale = rotemb(shorter, offset=(longer-shorter)//2 + 1)
    shorter_scale = scale
    assert torch.all(shorter_scale == longer_scale[-shorter:])
    ic(scale[:,0])


if __name__ == "__main__":
    main()
