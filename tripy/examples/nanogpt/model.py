
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Optional

import tripy as tp
from tripy import utils


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
    dtype: "tripy.datatype" = tp.float32
    quant_mode: str = None


def linear_layer(config: GPTConfig, in_feat, out_feat, bias):
    quant_kwargs = {}
    if config.quant_mode == "int8-weight-only":
        quant_kwargs["quant_dtype"] = tp.int8
        quant_kwargs["weight_quant_dim"] = 0
    elif config.quant_mode == "int4-weight-only":
        quant_kwargs["quant_dtype"] = tp.int4
        quant_kwargs["weight_quant_dim"] = None
    elif config.quant_mode == "fp8":
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
        self.zeros = tp.zeros((1, 1, self.seq_len, self.seq_len), dtype=config.dtype)

    def __call__(self, x: tp.Tensor):
        B, T, C = x.shape
        qkv = self.c_attn(x)  # (batch_size, seq_len, 3 * embedding_size)

        # WAR for better accuracy and avoid TRT compilation error in fp16
        if self.c_attn.quant_dtype in (tp.float8, tp.int4):
            qkv = tp.cast(qkv, tp.float32)

        q, k, v = tp.split(qkv, 3, dim=2)
        multi_head_output_shape = tp.concatenate(
            [B, T, tp.Tensor([self.num_heads]), tp.Tensor([self.embedding_size // self.num_heads])], dim=0
        )
        # WAR to prevent computing output rank in infer_rank for reshape, will be addressed by #228
        multi_head_output_shape.trace_tensor.shape = (4,)
        multi_heads = lambda x: tp.transpose(tp.reshape(x, multi_head_output_shape), 1, 2)

        q = multi_heads(q)
        k = multi_heads(k)
        v = multi_heads(v)

        k_t = tp.transpose(k, -2, -1)
        att = (q @ k_t) * (1.0 / math.sqrt(self.embedding_size // self.num_heads))
        att = tp.masked_fill(
            att,
            self.bias[:, :, :T, :T] == self.zeros[:, :, :T, :T],
            float("-inf"),
        )

        att = tp.softmax(att, dim=-1)

        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_size) -> (batch_size, num_heads, seq_len, head_size)
        out = att @ v
        out = tp.cast(out, x.dtype)
        out_shape = tp.concatenate([B, T, tp.Tensor([self.embedding_size])], dim=0)
        # WAR to prevent computing output rank in infer_rank for reshape, will be addressed by #228
        out_shape.trace_tensor.shape = (3,)
        out = tp.reshape(tp.transpose(out, 1, 2), out_shape)
        out = self.c_proj(out)  # (batch_size, seq_len, embedding_size)
        return out


class MLP(tp.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = linear_layer(config, config.embedding_size, 4 * config.embedding_size, config.bias)
        self.c_proj = linear_layer(config, 4 * config.embedding_size, config.embedding_size, config.bias)

    def __call__(self, x):
        x = self.c_fc(x)
        x = tp.gelu(x)
        x = self.c_proj(x)
        return x


class Block(tp.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = tp.LayerNorm(config.embedding_size)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = tp.LayerNorm(config.embedding_size)
        self.mlp = MLP(config)

    def __call__(self, x):
        x_ln1 = tp.cast(self.ln_1(tp.cast(x, self.ln_1.dtype)), x.dtype)
        x = x + self.attn(x_ln1)
        x_ln2 = tp.cast(self.ln_2(tp.cast(x, self.ln_2.dtype)), x.dtype)
        x = x + self.mlp(x_ln2)
        return x


class Transformer(tp.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = tp.Embedding(config.vocab_size, config.embedding_size, dtype=config.dtype)
        self.wpe = tp.Embedding(config.block_size, config.embedding_size, dtype=config.dtype)
        self.h = [Block(config) for _ in range(config.num_layers)]
        self.ln_f = tp.LayerNorm(config.embedding_size)
        self.pos = tp.reshape(tp.arange(0, config.seq_len, dtype=tp.int32), (1, config.seq_len))

    def __call__(self, idx):
        tok_emb = self.wte(idx)  # token embeddings of shape (batch_size, seq_len, embedding_size)
        pos_emb = self.wpe(self.pos[:, : idx.shape[1]])  # position embeddings of shape (seq_len, embedding_size)
        x = tok_emb + pos_emb  # (batch_size, seq_len, embedding_size)
        for block in self.h:
            x = block(x)
        x = tp.cast(self.ln_f(tp.cast(x, self.ln_f.dtype)), x.dtype)
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
        if config.quant_mode == "fp8":
            self.lm_head = linear_layer(config, config.embedding_size, config.vocab_size, bias=False)
        else:
            # lm_head is disabled for int8 quantization
            self.lm_head = tp.Linear(config.embedding_size, config.vocab_size, bias=False, dtype=config.dtype)

    def __call__(self, idx):
        x = self.transformer(idx)
        logits = self.lm_head(x)  # (batch_size, seq_len, embedding_size) -> (batch_size, seq_len, vocab_size)
        return logits
