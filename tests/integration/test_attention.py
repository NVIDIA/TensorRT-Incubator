from typing import Any
import numpy as np
import pytest
import math

import tripy


@pytest.mark.parametrize("bse", [(1, 10, 256), (2, 5, 256)])
@pytest.mark.parametrize("use_jit", [False, True])
def test_causal_self_attention(bse, use_jit):
    B, S, E = bse

    class CausalSelfAttention(tripy.nn.Module):
        def __init__(self):
            super().__init__()
            self.block_size = 1024
            self.n_head = 16
            self.n_embed = E
            self.c_attn = tripy.nn.Linear(self.n_embed, 3 * self.n_embed, bias=True)
            self.bias = tripy.tril(tripy.ones((self.block_size, self.block_size)))

        def __call__(self, x: tripy.Tensor) -> Any:
            attn = self.c_attn(x)

            # q, k, v = attn.split(self.n_embed, dim=2)
            q = attn[:, :, 0 : self.n_embed]
            k = attn[:, :, self.n_embed : self.n_embed * 2]
            v = attn[:, :, self.n_embed * 2 :]

            k = k.reshape((B, S, self.n_head, E // self.n_head)).transpose(1, 2)  # (B, nh, T, hs)
            q = k.reshape((B, S, self.n_head, E // self.n_head)).transpose(1, 2)  # (B, nh, T, hs)
            v = k.reshape((B, S, self.n_head, E // self.n_head)).transpose(1, 2)  # (B, nh, T, hs)

            k_t = k.transpose(-2, -1)
            att = (q @ k_t) * (1.0 / math.sqrt(E // self.n_head))
            att = att.masked_fill(self.bias[:S, :S] == tripy.Tensor(np.zeros((S, S), dtype=np.float32)), float("0"))

            # #82 will add softmax op.
            # att = att.softmax(dim=-1)

            out = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            out = out.transpose(1, 2).reshape((B, S, E))
            return out

    attn = CausalSelfAttention()
    if use_jit:
        attn = tripy.jit(attn)

    x = tripy.Tensor(np.random.rand(B, S, E).astype(np.float32), device=tripy.device("gpu"))
    attn(x).eval()

    # Enable comparing jit vs non-jit when softmax is enabled and weights can be initialized randomly.
    # np.testing.assert_array_equal(attn(x)[0].numpy(), jittd_attn(x)[0].numpy())
