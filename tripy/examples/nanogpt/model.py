#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import math
from dataclasses import dataclass

import nvtripy as tp


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    num_layers: int = 12
    num_heads: int = 12
    embedding_size: int = 768
    bias: bool = True
    seq_len: int = 1
    batch_size: int = 1
    dtype: "nvtripy.datatype" = tp.float32
    quant_mode: str = None


def linear_layer(config: GPTConfig, in_feat, out_feat, bias):
    quant_kwargs = {}
    if config.quant_mode == "int8-weight-only":
        quant_kwargs["quant_dtype"] = tp.int8
        quant_kwargs["weight_quant_dim"] = 0
    elif config.quant_mode == "int4-weight-only":
        quant_kwargs["quant_dtype"] = tp.int4
        quant_kwargs["weight_quant_dim"] = None
    elif config.quant_mode == "float8":
        quant_kwargs["quant_dtype"] = tp.float8
        quant_kwargs["weight_quant_dim"] = None

    return tp.Linear(
        in_feat,
        out_feat,
        bias=bias,
        dtype=config.dtype,
        **quant_kwargs,
    )


class CausalSelfAttention(tp.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embedding_size % config.num_heads == 0
        self.seq_len = config.seq_len
        self.batch_size = config.batch_size
        self.num_heads = config.num_heads
        self.embedding_size = config.embedding_size
        self.c_attn = linear_layer(config, config.embedding_size, 3 * config.embedding_size, config.bias)
        self.c_proj = linear_layer(config, config.embedding_size, config.embedding_size, config.bias)
        self.bias = tp.reshape(
            tp.tril(tp.ones((config.block_size, config.block_size), dtype=config.dtype)),
            (1, 1, config.block_size, config.block_size),
        )

    def forward(self, x: tp.Tensor):
        B, T = x.shape[0:2]
        qkv = self.c_attn(x)  # (batch_size, seq_len, 3 * embedding_size)

        # WAR for better accuracy and avoid TRT compilation error in fp16
        if self.c_attn.quant_dtype in (tp.float8, tp.int4):
            qkv = tp.cast(qkv, tp.float32)

        q, k, v = tp.split(qkv, 3, dim=2)
        multi_head_output_shape = [B, T, self.num_heads, self.embedding_size // self.num_heads]
        multi_heads = lambda x: tp.transpose(tp.reshape(x, multi_head_output_shape), 1, 2)

        q = multi_heads(q)
        k = multi_heads(k)
        v = multi_heads(v)

        k_t = tp.transpose(k, -2, -1)
        att = (q @ k_t) * (1.0 / math.sqrt(self.embedding_size // self.num_heads))
        att = tp.masked_fill(att, self.bias[:, :, :T, :T] == 0, float("-inf"))

        att = tp.softmax(att, dim=-1)

        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_size) -> (batch_size, num_heads, seq_len, head_size)
        out = att @ v
        out = tp.cast(out, x.dtype)
        out = tp.reshape(tp.transpose(out, 1, 2), [B, T, self.embedding_size])
        out = self.c_proj(out)  # (batch_size, seq_len, embedding_size)
        return out


class MLP(tp.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = linear_layer(config, config.embedding_size, 4 * config.embedding_size, config.bias)
        self.c_proj = linear_layer(config, 4 * config.embedding_size, config.embedding_size, config.bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = tp.gelu(x)
        x = self.c_proj(x)
        return x


class Block(tp.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = tp.LayerNorm(config.embedding_size, dtype=config.dtype)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = tp.LayerNorm(config.embedding_size, dtype=config.dtype)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(tp.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len
        self.wte = tp.Embedding(config.vocab_size, config.embedding_size, dtype=config.dtype)
        self.wpe = tp.Embedding(config.block_size, config.embedding_size, dtype=config.dtype)
        self.h = tp.Sequential(*[Block(config) for _ in range(config.num_layers)])
        self.ln_f = tp.LayerNorm(config.embedding_size, dtype=config.dtype)

    def forward(self, idx):
        tok_emb = self.wte(idx)  # token embeddings of shape (batch_size, seq_len, embedding_size)
        pos = tp.unsqueeze(tp.arange(self.seq_len, dtype=tp.int32)[: idx.shape[1]], 0)
        pos_emb = self.wpe(pos)  # position embeddings of shape (seq_len, embedding_size)
        x = tok_emb + pos_emb  # (batch_size, seq_len, embedding_size)
        x = self.h(x)
        x = self.ln_f(x)
        return x


class GPT(tp.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert (
            config.seq_len <= config.block_size
        ), f"Cannot forward sequence of length {config.seq_len}, block size is only {config.block_size}"

        self.transformer = Transformer(config)

        if config.quant_mode == "float8":
            self.lm_head = linear_layer(config, config.embedding_size, config.vocab_size, bias=False)
        else:
            # Quantization is disabled for `lm_head` except for FP8.
            self.lm_head = tp.Linear(config.embedding_size, config.vocab_size, bias=False, dtype=config.dtype)

    def forward(self, idx):
        x = self.transformer(idx)
        logits = self.lm_head(x)  # (batch_size, seq_len, embedding_size) -> (batch_size, seq_len, vocab_size)
        return logits
