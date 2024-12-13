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

from functools import partial
from typing import Tuple, Type

import tripy as tp
from tripy import Tensor
from sam2.modeling.sam2_utils import MLP, scaled_dot_product_attention
from sam2.modeling.position_encoding import apply_rotary_enc, compute_axial_cis


class TwoWayTransformer(tp.Module):

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[tp.Module] = tp.relu,
        attention_downsample_rate: int = 2,
        dtype=tp.float32,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = []

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                    dtype=dtype,
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
            dtype=dtype,
        )
        self.norm_final_attn = tp.LayerNorm(embedding_dim)

    def __call__(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        return self.forward(image_embedding, image_pe, point_embedding)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (tp.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (tp.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (tp.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          tp.Tensor: the processed point_embedding
          tp.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = tp.permute(tp.flatten(image_embedding, 2), (0, 2, 1))
        image_pe = tp.permute(tp.flatten(image_pe, 2), (0, 2, 1))

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = tp.cast(
            self.norm_final_attn(tp.cast(queries, self.norm_final_attn.dtype)),
            queries.dtype,
        )
        # queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(tp.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[tp.Module] = tp.relu,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        dtype=tp.float32,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads, dtype=dtype)
        self.norm1 = tp.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
            dtype=dtype,
        )
        self.norm2 = tp.LayerNorm(embedding_dim)

        self.mlp = MLP(
            embedding_dim,
            mlp_dim,
            embedding_dim,
            num_layers=2,
            activation=activation,
            dtype=dtype,
        )
        self.norm3 = tp.LayerNorm(embedding_dim)

        self.norm4 = tp.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
            dtype=dtype,
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def __call__(self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor) -> Tuple[Tensor, Tensor]:
        return self.forward(queries, keys, query_pe, key_pe)

    def forward(self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out

        queries = tp.cast(self.norm1(tp.cast(queries, self.norm1.dtype)), queries.dtype)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out

        queries = tp.cast(self.norm2(tp.cast(queries, self.norm2.dtype)), queries.dtype)
        # queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = tp.cast(self.norm3(tp.cast(queries, self.norm3.dtype)), queries.dtype)
        # queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = tp.cast(self.norm4(tp.cast(keys, self.norm4.dtype)), keys.dtype)
        # keys = self.norm4(keys)

        return queries, keys


class Attention(tp.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
        dtype=tp.float32,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        self.q_proj = tp.Linear(embedding_dim, self.internal_dim, dtype=dtype)
        self.k_proj = tp.Linear(self.kv_in_dim, self.internal_dim, dtype=dtype)
        self.v_proj = tp.Linear(self.kv_in_dim, self.internal_dim, dtype=dtype)
        self.out_proj = tp.Linear(self.internal_dim, embedding_dim, dtype=dtype)

        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape[0], x.shape[1], x.shape[2]
        x = tp.reshape(x, [b, n, num_heads, c // num_heads])
        return tp.transpose(x, 1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_head, n_token, c_per_head = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = tp.transpose(x, 1, 2)
        return tp.reshape(x, [b, n_token, n_head * c_per_head])  # B x N_tokens x C

    def __call__(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return self.forward(q, k, v)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        out = scaled_dot_product_attention(q, k, v, embedding_dim=k.shape[-1])
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class RoPEAttention(Attention):
    """Attention with rotary position encoding."""

    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        # whether to repeat q rope to match k length
        # this is needed for cross-attention to memories
        rope_k_repeat=False,
        feat_sizes=(32, 32),  # [w, h] for stride 16 feats at 512 resolution
        dtype="float32",
        **kwargs,
    ):
        self.dtype = getattr(tp, dtype)
        super().__init__(*args, dtype=self.dtype, **kwargs)
        self.compute_cis = partial(compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta)
        self.rope_k_repeat = rope_k_repeat

    def __call__(self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: tp.Tensor) -> Tensor:
        return self.forward(q, k, v, num_k_exclude_rope)

    def forward(self, q: tp.Tensor, k: tp.Tensor, v: tp.Tensor, num_k_exclude_rope: tp.Tensor) -> tp.Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Apply rotary position encoding
        # w = h = tp.DimensionSize(tp.cast(tp.sqrt(tp.cast(q.shape[-2], tp.float32)), tp.int32))  # DDS?
        w = h = tp.DimensionSize(64)  # Current demo always uses 64.
        freqs_cis = self.compute_cis(end_x=w, end_y=h)
        self.freqs_cis = tp.cast(freqs_cis, self.dtype)

        num_k_rope = k.shape[-2] - num_k_exclude_rope
        q, new_k = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )
        k = tp.concatenate([new_k, k[:, :, num_k_rope:, :]], dim=-2)

        # Attention
        out = scaled_dot_product_attention(q, k, v, embedding_dim=k.shape[-1])
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
