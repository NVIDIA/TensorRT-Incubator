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

import tripy as tp

import tripy as tp
from dataclasses import dataclass

from examples.diffusion.helper import scaled_dot_product_attention

@dataclass
class CLIPConfig:
    vocab_size: int = 49408
    embedding_size: int = 768
    num_heads: int = 12
    max_seq_len: int = 77
    num_hidden_layers: int = 12

class CLIPMLP(tp.Module):
    def __init__(self, config: CLIPConfig):
        self.fc1 = tp.Linear(config.embedding_size, config.embedding_size * 4)
        self.fc2 = tp.Linear(config.embedding_size * 4, config.embedding_size)

    def __call__(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = tp.sigmoid(1.702 * hidden_states) * hidden_states  # quick GELU
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPAttention(tp.Module):
    def __init__(self, config: CLIPConfig):
        self.embed_dim = config.embedding_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.k_proj = tp.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = tp.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = tp.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = tp.Linear(self.embed_dim, self.embed_dim)

    def __call__(self, hidden_states, causal_attention_mask):
        bsz, tgt_len, embed_dim = hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[2]
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        q, k, v = [
            tp.transpose(
                tp.reshape(x, (bsz, tgt_len, self.num_heads, self.head_dim)),
                1,
                2,
            )
            for x in (q, k, v)
        ]
        attn_output = scaled_dot_product_attention(
            q, k, v, embedding_dim=self.head_dim, attn_mask=causal_attention_mask
        )
        out = self.out_proj(tp.reshape(tp.transpose(attn_output, 1, 2), (bsz, tgt_len, embed_dim)))
        return out


class CLIPEncoderLayer(tp.Module):
    def __init__(self, config: CLIPConfig):
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = tp.LayerNorm(config.embedding_size)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = tp.LayerNorm(config.embedding_size)

    def __call__(self, hidden_states, causal_attention_mask):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, causal_attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(tp.Module):
    def __init__(self, config: CLIPConfig):
        self.layers = [CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, hidden_states, causal_attention_mask):
        for l in self.layers:
            hidden_states = l(hidden_states, causal_attention_mask)
        return hidden_states


class CLIPTextEmbeddings(tp.Module):
    def __init__(self, config: CLIPConfig):
        self.token_embedding = tp.Embedding(config.vocab_size, config.embedding_size)
        self.position_embedding = tp.Embedding(config.max_seq_len, config.embedding_size)

    def __call__(self, input_ids, position_ids):
        return self.token_embedding(input_ids) + self.position_embedding(position_ids)


class CLIPTextTransformer(tp.Module):
    def __init__(self, config: CLIPConfig):
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = tp.LayerNorm(config.embedding_size)
        self.max_seq_len = config.max_seq_len

    def __call__(self, input_ids):
        x = self.embeddings(input_ids, tp.reshape(tp.iota((input_ids.shape[1],), dtype=tp.int32), (1, -1)))
        x = self.encoder(x, tp.triu(tp.full((1, 1, self.max_seq_len, self.max_seq_len), float("-inf")), 1))
        return self.final_layer_norm(x)