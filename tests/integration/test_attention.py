from typing import Any
import numpy as np
import pytest
import math

import tripy


@pytest.mark.skip(reason="nn.Linear, softmax, reshape, size ops missing from tripy.")
@pytest.mark.parametrize(
    "use_jit",
    [False, True],
)
def test_causal_self_attention(self, use_jit):
    class CausalSelfAttention(tripy.nn.Module):
        def __init__(self):
            super().__init__()
            self.block_size = 1024
            self.n_head = 16
            self.n_embed = 1024
            self.c_attn = tripy.nn.Linear(self.n_embed, 3 * self.n_embed, bias=True)
            self.bias = tripy.tril(tripy.ones(1, 1, self.block_size, self.block_size))

        def __call__(self, x: tripy.Tensor) -> Any:
            B, T, C = x.size()
            attn = self.c_attn(x)
            q, k, v = attn.split(self.n_embed, dim=2)
            k = k.reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            q = k.reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            v = k.reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

            k_t = k.transpose(-2, -1)
            att = (q @ k) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = att.softmax(dim=-1)
            out = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            out = out.transpose(1, 2).reshape(B, T, C)
            return out

    attn = CausalSelfAttention()
    if use_jit:
        attn = tripy.jit(attn)

    x = tripy.Tensor(np.random.rand(2, 10, 128).astype(np.float32))
    print(attn(x))
