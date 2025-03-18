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

from functools import partial
from typing import Callable, List, Tuple, Union

import nvtripy as tp
from sam2.modeling.backbones.utils import (
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
        dtype: tp.dtype = tp.float32,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        self.qkv = tp.Linear(dim, dim_out * 3, dtype=dtype)
        self.proj = tp.Linear(dim_out, dim_out, dtype=dtype)

    def forward(self, x):
        B, H, W = x.shape[0:3]
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = tp.reshape(self.qkv(x), (B, H * W, 3, self.num_heads, -1))
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(tp.reshape(q, (B, H, W, -1)), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = tp.reshape(q, (B, H * W, self.num_heads, -1))

        x = scaled_dot_product_attention(
            tp.transpose(q, 1, 2),
            tp.transpose(k, 1, 2),
            tp.transpose(v, 1, 2),
        )
        # Transpose back
        x = tp.transpose(x, 1, 2)
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
        pad_size: int = 0,
        unpad_size: int = 0,
        dtype: tp.dtype = tp.float32,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(tp, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size
        self.pad_size = (pad_size, pad_size)
        self.unpad_size = (unpad_size, unpad_size)

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = partial(tp.maxpool, kernel_dims=q_stride, stride=q_stride)

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
            dtype=dtype,
        )

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
            dtype=dtype,
        )

        if dim != dim_out:
            self.proj = tp.Linear(dim, dim_out, dtype=dtype)

    def forward(self, x):

        def call_norm(x, norm):
            x_dtype = x.dtype
            x = tp.cast(x, tp.float32)
            x = norm(x)
            return tp.cast(x, x_dtype)

        shortcut = x  # B, H, W, C
        x = call_norm(x, self.norm1)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1:3]
            x, pad_hw = window_partition(x, window_size, self.pad_size)

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            def mod_int(x, y):
                return x - (x / y) * y

            pad_h = mod_int((window_size - mod_int(H, window_size)), window_size)
            pad_w = mod_int((window_size - mod_int(W, window_size)), window_size)
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, self.unpad_size)

        x = shortcut + x
        # MLP
        t = call_norm(x, self.norm2)
        x = x + self.mlp(t)
        return x


class Hiera(tp.Module):

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
        block_pad_size: List[int] = [],
        block_unpad_size: List[int] = [],
        return_interm_layers=True,  # return feats from every stage
        dtype: str = "float32",
    ):
        super().__init__()

        self.dtype = dtype
        tp_dtype = getattr(tp, dtype)
        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.block_pad_size = block_pad_size if block_pad_size else [0] * depth
        self.block_unpad_size = block_unpad_size if block_unpad_size else [0] * depth
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(embed_dim=embed_dim, dtype=tp_dtype)
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = tp.zeros((1, embed_dim, *self.window_pos_embed_bkg_spatial_size), dtype=tp_dtype)
        self.pos_embed_window = tp.zeros((1, embed_dim, self.window_spec[0], self.window_spec[0]), dtype=tp_dtype)
        self.pos_embed_torch = None

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
                pad_size=self.block_pad_size[i],
                unpad_size=self.block_unpad_size[i],
                dtype=tp_dtype,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

    def generate_static_pos_embed(self, hw: Tuple[int, int]):
        import torch
        import torch.nn.functional as F

        h, w = hw
        window_embed = torch.from_dlpack(self.pos_embed_window)
        pos_embed = F.interpolate(torch.from_dlpack(self.pos_embed), size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile([x // y for x, y in zip(pos_embed.shape, window_embed.shape)])
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        self.pos_embed_torch = pos_embed.contiguous()

    def forward(self, x: tp.Tensor):
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # Add pos embed
        x = x + tp.Tensor(self.pos_embed_torch)

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (i in self.stage_ends and self.return_interm_layers):
                # [1, 7, 43, 47]
                feats = tp.permute(x, (0, 3, 1, 2))
                outputs.append(feats)

        return outputs
