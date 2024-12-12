# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling SAM2 with Tripy or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional

from sam2.modeling.sam.transformer import RoPEAttention
from sam2.modeling.sam2_utils import get_activation_fn, get_clones

import tripy as tp


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

        self.norm1 = tp.LayerNorm(d_model)
        self.norm2 = tp.LayerNorm(d_model)
        self.norm3 = tp.LayerNorm(d_model)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)

        # Where to add pos enc
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos):
        # Self-Attention
        tgt2 = tp.cast(self.norm1(tp.cast(tgt, self.norm1.dtype)), self.dtype)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2, num_k_exclude_rope=tp.Tensor([0]))
        tgt = tgt + tgt2
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        kwds = {}

        # Cross-Attention
        tgt2 = tp.cast(self.norm2(tp.cast(tgt, self.norm2.dtype)), self.dtype)

        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            num_k_exclude_rope=num_k_exclude_rope,
        )
        tgt = tgt + tgt2
        return tgt

    def __call__(
        self,
        tgt,
        memory,
        pos: Optional[tp.Tensor] = None,
        query_pos: Optional[tp.Tensor] = None,
        num_k_exclude_rope: Optional[tp.Tensor] = None,
    ) -> tp.Tensor:

        # Self-Attn, Cross-Attn
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        # MLP
        tgt2 = tp.cast(self.norm3(tp.cast(tgt, self.norm3.dtype)), self.dtype)

        tgt2 = self.linear2(self.activation(self.linear1(tgt2)))
        tgt = tgt + tgt2
        return tgt


class MemoryAttention(tp.Module):

    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: tp.Module,
        num_layers: int,
        batch_first: bool = True,  # Do layers expect batch first input?
        dtype="float32",
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = tp.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first
        self.dtype = getattr(tp, dtype)

    def __call__(
        self,
        curr: tp.Tensor,  # self-attention inputs
        memory: tp.Tensor,  # cross-attention inputs
        curr_pos: Optional[tp.Tensor] = None,  # pos_enc for self-attention inputs
        memory_pos: Optional[tp.Tensor] = None,  # pos_enc for cross-attention inputs
        num_obj_ptr_tokens: Optional[tp.Tensor] = None,  # number of object pointer *tokens*
    ):

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

        normed_output = tp.cast(self.norm(tp.cast(output, self.norm.dtype)), self.dtype)

        if self.batch_first:
            # Convert back to seq first
            normed_output = tp.transpose(normed_output, 0, 1)

        return normed_output
