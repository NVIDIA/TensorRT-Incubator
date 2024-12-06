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

import math
from typing import Tuple

import tripy as tp

from sam2.modeling.sam2_utils import LayerNorm2d


class Dummy(tp.Module):
    def __init__(self):
        pass

    def __call__(self, x):
        return x


class MaskDownSampler(tp.Module):
    """
    Progressively downsample a mask by total_stride.
    """

    def __init__(
        self,
        embed_dim=256,
        kernel_size=4,
        stride=4,
        padding=0,
        total_stride=16,
        activation=tp.gelu,
        dtype="float32",
    ):
        super().__init__()
        self.dtype = getattr(tp, dtype)
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        assert stride**num_layers == total_stride
        self.encoder = []
        mask_in_chans, mask_out_chans = 1, 1
        for _ in range(num_layers):
            mask_out_chans = mask_in_chans * (stride**2)
            self.encoder.append(
                tp.Conv(
                    mask_in_chans,
                    mask_out_chans,
                    kernel_dims=(kernel_size, kernel_size),
                    stride=(stride, stride),
                    padding=((padding, padding), (padding, padding)),
                    dtype=self.dtype,
                )
            )
            self.encoder.append(LayerNorm2d(mask_out_chans))
            self.encoder.append(activation)
            mask_in_chans = mask_out_chans

        self.encoder.append(tp.Conv(mask_out_chans, embed_dim, kernel_dims=(1, 1), dtype=self.dtype))

    def __call__(self, x):
        for l in self.encoder:
            x = l(x)
        return x


class CXBlock(tp.Module):
    r"""ConvNeXt Block.
    DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    """

    def __init__(
        self,
        dim,
        kernel_size=7,
        padding=3,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        use_dwconv=True,
        dtype="float32",
    ):
        super().__init__()
        self.dtype = getattr(tp, dtype)
        self.dwconv = tp.Conv(
            dim,
            dim,
            kernel_dims=(kernel_size, kernel_size),
            padding=((padding, padding), (padding, padding)),
            groups=dim if use_dwconv else 1,
            dtype=self.dtype,
        )  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = tp.Linear(dim, 4 * dim, dtype=self.dtype)  # pointwise/1x1 convs, implemented with linear layers
        self.act = tp.gelu
        self.pwconv2 = tp.Linear(4 * dim, dim, dtype=self.dtype)
        self.gamma = tp.ones((dim,), dtype=self.dtype) * layer_scale_init_value

        self.drop_path = Dummy()

    def __call__(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = tp.permute(x, (0, 2, 3, 1))  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = tp.permute(x, (0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Fuser(tp.Module):
    def __init__(self, layer, num_layers, dim=None, input_projection=False, dtype="float32"):
        super().__init__()
        self.dtype = getattr(tp, dtype)
        self.proj = Dummy()
        self.layers = [layer for i in range(num_layers)]

        if input_projection:
            self.proj = tp.Conv(dim, dim, kernel_dims=(1, 1), dtype=self.dtype)

    def __call__(self, x):
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


class MemoryEncoder(tp.Module):
    def __init__(
        self, out_dim, mask_downsampler, fuser, position_encoding, in_dim=256, dtype="float32"  # in_dim of pix_feats
    ):
        super().__init__()
        self.dtype = getattr(tp, dtype)

        self.mask_downsampler = mask_downsampler
        self.pix_feat_proj = tp.Conv(in_dim, in_dim, kernel_dims=(1, 1), dtype=self.dtype)
        self.fuser = fuser
        self.position_encoding = position_encoding
        self.out_proj = Dummy()
        if out_dim != in_dim:
            self.out_proj = tp.Conv(in_dim, out_dim, kernel_dims=(1, 1), dtype=self.dtype)

    def __call__(
        self,
        pix_feat: tp.Tensor,
        masks: tp.Tensor,
        skip_mask_sigmoid: bool = False,
    ) -> Tuple[tp.Tensor, tp.Tensor]:
        if not skip_mask_sigmoid:
            masks = tp.sigmoid(masks)
        masks = self.mask_downsampler(masks)

        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)

        pos = tp.cast(self.position_encoding(x), x.dtype)

        return x, pos
