from typing import Any
import numpy as np
import pytest
import math

import tripy.common.datatype
import tripy


@pytest.mark.skip(reason="Dynamic shape is not working with MLIR backend yet.")
@pytest.mark.parametrize(
    "use_jit",
    [False, True],
)
def test_causal_self_attention(self, use_jit):
    class CausalSelfAttention(tripy.nn.Module):
        def __init__(self):
            super().__init__()
            self.n_head = 16
            self.n_embed = 1024
            self.c_attn = tripy.nn.Linear(self.n_embed, 3 * self.n_embed, bias=True)

        def __call__(self, x: tripy.Tensor) -> Any:
            from tripy.frontend.ops import transpose, softmax

            B, T, C = x.size()
            attn = self.c_attn(x)
            q, k, v = attn.split(self.n_embed, dim=2)
            k = transpose(k.view(B, T, self.n_head, C // self.n_head), 1, 2)  # (B, nh, T, hs)
            q = transpose(k.view(B, T, self.n_head, C // self.n_head), 1, 2)  # (B, nh, T, hs)
            v = transpose(k.view(B, T, self.n_head, C // self.n_head), 1, 2)  # (B, nh, T, hs)

            k_t = transpose(k, -2, -1)
            att = (q @ k) * (1.0 / math.sqrt(k.size(-1)))
            att = softmax(att, dim=-1)
            out = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            out = transpose(out, 1, 2).view(B, T, C)

            return out

    attn = CausalSelfAttention()
    if use_jit:
        attn = tripy.jit(attn)

    x = tripy.Tensor(np.random.rand(2, 10, 128).astype(np.float32))
    print(attn(x))
