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
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Callable, List, Tuple, Union

import tripy as tp
from backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)
from sam2.modeling.sam2_utils import MLP, scaled_dot_product_attention


def do_pool(x, pool, norm=None):
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = tp.permute(x, (0, 3, 1, 2))
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = tp.permute(x, (0, 2, 3, 1))
    if norm:
        x = norm(x)

    return x


class MultiScaleAttention(tp.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: tp.Module = None,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        self.qkv = tp.Linear(dim, dim_out * 3)
        self.proj = tp.Linear(dim_out, dim_out)

    def forward(self, x):
        B, H, W = x.shape[0:3]
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = tp.reshape(self.qkv(x), (B, H * W, 3, self.num_heads, -1))
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = tp.split(qkv, 3, dim=2)

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(tp.reshape(q, (B, H, W, -1)), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = tp.reshape(q, (B, H * W, self.num_heads, -1))

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = scaled_dot_product_attention(
            tp.transpose(q, (1, 2)),
            tp.transpose(k, (1, 2)),
            tp.transpose(v, (1, 2)),
        )
        # Transpose back
        x = tp.transpose(x, (1, 2))
        x = tp.reshape(x, (B, H, W, -1))

        x = self.proj(x)

        return x


class MultiScaleBlock(tp.Module):

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        norm_layer: Union[tp.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: Callable = tp.gelu,
        window_size: int = 0,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(tp, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            # q_stride is (2, 2)
            self.pool = partial(tp.maxpool, kernel_dims=q_stride, stride=q_stride)

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )

        if dim != dim_out:
            self.proj = tp.Linear(dim, dim_out)

    def forward(self, x):
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1:3]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + x
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class Hiera(tp.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989
    """

    def __init__(
        self,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
        return_interm_layers=True,  # return feats from every stage
    ):
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
        )
        # Which blocks have global att?
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = tp.Parameter(tp.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size))
        self.pos_embed_window = tp.Parameter(tp.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0]))

        cur_stage = 1
        self.blocks = []

        for i in range(depth):
            dim_out = embed_dim
            # lags by a block, so first block of
            # next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

    def _get_pos_embed(self, hw: Tuple[int, int]) -> tp.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed_shape = (self.pos_embed.shape[0], self.pos_embed.shape[1], h, w)
        pos_embed = tp.resize(self.pos_embed, mode="cubic", output_shape=pos_embed_shape)
        # WAR: tp.repeat twice
        window_embed = tp.repeat(window_embed, h // self.window_spec[0], dim=-2)
        window_embed = tp.repeat(window_embed, w // self.window_spec[0], dim=-1)
        pos_embed = pos_embed + window_embed
        pos_embed = tp.permute(pos_embed, (0, 2, 3, 1))
        return pos_embed

    def forward(self, x: tp.Tensor) -> List[tp.Tensor]:
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # Add pos embed
        h, w = x.shape[1:3]
        x = x + self._get_pos_embed((h, w))

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (i in self.stage_ends and self.return_interm_layers):
                feats = tp.permute(x, (0, 3, 1, 2))
                outputs.append(feats)

        return outputs
