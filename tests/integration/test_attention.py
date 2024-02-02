from typing import Any
import numpy as np
import pytest
import math

import tripy


@pytest.mark.skip(reason="Testing with random values is not stable.")
@pytest.mark.parametrize("bse", [(1, 10, 256), (2, 5, 256)])
@pytest.mark.skip("Intermittenly fails due to numerical instability")
def test_causal_self_attention(bse):
    B, S, E = bse

    class CausalSelfAttention(tripy.nn.Module):
        def __init__(self):
            super().__init__()
            self.block_size = 1024
            self.n_head = 16
            self.n_embed = E
            self.c_attn = tripy.nn.Linear(self.n_embed, 3 * self.n_embed, bias=True)
            self.c_proj = tripy.nn.Linear(self.n_embed, self.n_embed, bias=True)
            self.bias = tripy.ones((self.block_size, self.block_size)).tril()

        def __call__(self, x: tripy.Tensor) -> Any:
            attn = self.c_attn(x)

            # q, k, v = attn.split(self.n_embed, dim=2)
            q = attn[:, :, 0 : self.n_embed]
            k = attn[:, :, self.n_embed : self.n_embed * 2]
            v = attn[:, :, self.n_embed * 2 :]

            k = k.reshape((B, S, self.n_head, E // self.n_head)).transpose(1, 2)  # (B, nh, T, hs)
            q = q.reshape((B, S, self.n_head, E // self.n_head)).transpose(1, 2)  # (B, nh, T, hs)
            v = v.reshape((B, S, self.n_head, E // self.n_head)).transpose(1, 2)  # (B, nh, T, hs)

            k_t = k.transpose(-2, -1)
            att = (q @ k_t) * (1.0 / math.sqrt(E // self.n_head))
            att = att.masked_fill(self.bias[:S, :S] == tripy.Tensor(np.zeros((S, S), dtype=np.float32)), float("-inf"))
            att = tripy.nn.softmax(att, dim=-1)
            out = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            out = out.transpose(1, 2).reshape((B, S, E))
            out = self.c_proj(out)
            return out

    attn = CausalSelfAttention()
    jitted_attn = tripy.jit(attn)

    x = tripy.Tensor(np.random.rand(B, S, E).astype(np.float32), device=tripy.device("gpu"))
    attn(x).eval()

    # TODO: enable random initialized weights
    np.testing.assert_allclose(attn(x)[0].numpy(), jitted_attn(x)[0].numpy())
