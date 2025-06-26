# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling SAM2 with Tripy or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
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

from typing import List, Optional

import nvtripy as tp
from sam2.modeling.sam2_utils import get_activation_fn
from sam2.modeling.sam.transformer import RoPEAttention


class MemoryAttentionLayer(tp.Module):
    def __init__(
        self,
        activation: str,
        cross_attention: tp.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: tp.Module,
        dtype: "float32",
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention
        self.dtype = getattr(tp, dtype)

        # Implementation of Feedforward model
        self.linear1 = tp.Linear(d_model, dim_feedforward, dtype=self.dtype)
        self.linear2 = tp.Linear(dim_feedforward, d_model, dtype=self.dtype)

        self.norm1 = tp.LayerNorm(d_model, dtype=self.dtype)
        self.norm2 = tp.LayerNorm(d_model, dtype=self.dtype)
        self.norm3 = tp.LayerNorm(d_model, dtype=self.dtype)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)

        # Where to add pos enc
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2, num_k_exclude_rope=0)
        tgt = tgt + tgt2
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        # Cross-Attention
        tgt2 = self.norm2(tgt)

        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            num_k_exclude_rope=num_k_exclude_rope,
        )
        tgt = tgt + tgt2
        return tgt

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[tp.Tensor] = None,
        query_pos: Optional[tp.Tensor] = None,
        num_k_exclude_rope: Optional[tp.types.IntLike] = None,
    ) -> tp.Tensor:

        # Self-Attn, Cross-Attn
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        # MLP
        tgt2 = self.norm3(tgt)

        tgt2 = self.linear2(self.activation(self.linear1(tgt2)))
        tgt = tgt + tgt2
        return tgt


class MemoryAttention(tp.Module):
    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        num_layers: int,
        activation: str,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        sa_rope_theta: float,
        sa_feat_sizes: List[int],
        sa_embedding_dim: int,
        sa_num_heads: int,
        sa_downsample_rate: int,
        sa_dropout: float,
        ca_rope_theta: float,
        ca_feat_sizes: List[int],
        ca_rope_k_repeat: bool,
        ca_embedding_dim: int,
        ca_num_heads: int,
        ca_downsample_rate: int,
        ca_dropout: float,
        ca_kv_in_dim: int,
        batch_first: bool = True,
        dtype="float32",
    ):
        super().__init__()
        self.dtype = getattr(tp, dtype)

        self.d_model = d_model
        self.num_layers = num_layers
        self.norm = tp.LayerNorm(d_model, self.dtype)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first
        self.layers = []
        for _ in range(num_layers):
            self_attn = RoPEAttention(
                sa_embedding_dim,
                sa_num_heads,
                sa_downsample_rate,
                sa_dropout,
                rope_theta=sa_rope_theta,
                feat_sizes=sa_feat_sizes,
                dtype=dtype,
            )
            cross_attn = RoPEAttention(
                ca_embedding_dim,
                ca_num_heads,
                ca_downsample_rate,
                ca_dropout,
                ca_kv_in_dim,
                rope_theta=ca_rope_theta,
                rope_k_repeat=ca_rope_k_repeat,
                feat_sizes=ca_feat_sizes,
                dtype=dtype,
            )
            memory_attn_layer = MemoryAttentionLayer(
                activation=activation,
                cross_attention=cross_attn,
                d_model=d_model,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                pos_enc_at_attn=pos_enc_at_attn,
                pos_enc_at_cross_attn_keys=pos_enc_at_cross_attn_keys,
                pos_enc_at_cross_attn_queries=pos_enc_at_cross_attn_queries,
                self_attention=self_attn,
                dtype=dtype,
            )
            self.layers.append(memory_attn_layer)

    def forward(
        self,
        curr: tp.Tensor,  # self-attention inputs
        memory: tp.Tensor,  # cross-attention inputs
        curr_pos: Optional[tp.Tensor] = None,  # pos_enc for self-attention inputs
        memory_pos: Optional[tp.Tensor] = None,  # pos_enc for cross-attention inputs
        num_obj_ptr_tokens: Optional[tp.Tensor] = None,  # number of object pointer *tokens*
    ):
        # TODO (#594): Remove this hack once we are able to pass in DimensionSizes directly:
        num_obj_ptr_tokens = num_obj_ptr_tokens.shape[0]
        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            # Convert to batch first
            output = tp.transpose(output, 0, 1)
            memory = tp.transpose(memory, 0, 1)
            if curr_pos is not None:
                curr_pos = tp.transpose(curr_pos, 0, 1)
            if memory_pos is not None:
                memory_pos = tp.transpose(memory_pos, 0, 1)

        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                **kwds,
            )

        normed_output = self.norm(output)

        if self.batch_first:
            # Convert back to seq first
            normed_output = tp.transpose(normed_output, 0, 1)

        return normed_output
