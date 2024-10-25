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

import math
from typing import Tuple

import tripy as tp

from sam2.modeling.tripy.sam2_utils import LayerNorm2d
from sam2.modeling.tripy.sam2_utils import get_clones


class Dummy(tp.Module):
    def __init__(self):
        pass

    def __call__(self, x):
        return x


class MaskDownSampler(tp.Module):
    """
    Progressively downsample a mask by total_stride, each time by stride.
    Note that LayerNorm is applied per *token*, like in ViT.

    With each downsample (by a factor stride**2), channel capacity increases by the same factor.
    In the end, we linearly project to embed_dim channels.
    """

    def __init__(
        self,
        embed_dim=256,
        kernel_size=4,
        stride=4,
        padding=0,
        total_stride=16,
        activation=tp.gelu,
    ):
        super().__init__()
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
                )
            )
            self.encoder.append(LayerNorm2d(mask_out_chans))
            self.encoder.append(activation)
            mask_in_chans = mask_out_chans

        self.encoder.append(tp.Conv(mask_out_chans, embed_dim, kernel_dims=(1, 1)))

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        for l in self.encoder:
            x = l(x)

        return x


class CXBlock(tp.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim,
        kernel_size=7,
        padding=3,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        use_dwconv=True,
    ):
        super().__init__()
        self.dwconv = tp.Conv(
            dim,
            dim,
            kernel_dims=(kernel_size, kernel_size),
            padding=((padding, padding), (padding, padding)),
            groups=dim if use_dwconv else 1,
        )  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = tp.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = tp.gelu
        self.pwconv2 = tp.Linear(4 * dim, dim)
        self.gamma = tp.ones((dim,)) * layer_scale_init_value

        # self.gamma = (
        #     nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        #     if layer_scale_init_value > 0
        #     else None
        # )
        self.drop_path = Dummy()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
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
    def __init__(self, layer, num_layers, dim=None, input_projection=False):
        super().__init__()
        self.proj = Dummy()
        self.layers = get_clones(layer, num_layers)

        if input_projection:
            self.proj = tp.Conv(dim, dim, kernel_dims=(1, 1))

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # normally x: (N, C, H, W)
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


class MemoryEncoder(tp.Module):
    def __init__(
        self,
        out_dim,
        mask_downsampler,
        fuser,
        position_encoding,
        in_dim=256,  # in_dim of pix_feats
    ):
        super().__init__()

        self.mask_downsampler = mask_downsampler

        self.pix_feat_proj = tp.Conv(in_dim, in_dim, kernel_dims=(1, 1))
        self.fuser = fuser
        self.position_encoding = position_encoding
        self.out_proj = Dummy()
        if out_dim != in_dim:
            self.out_proj = tp.Conv(in_dim, out_dim, kernel_dims=(1, 1))

    def __call__(
        self,
        pix_feat: tp.Tensor,
        masks: tp.Tensor,
        skip_mask_sigmoid: bool = False,
    ):
        return self.forward(pix_feat, masks, skip_mask_sigmoid)

    def forward(
        self,
        pix_feat: tp.Tensor,
        masks: tp.Tensor,
        skip_mask_sigmoid: bool = False,
    ) -> Tuple[tp.Tensor, tp.Tensor]:
        ## Process masks
        # sigmoid, so that less domain shift from gt masks which are bool
        if not skip_mask_sigmoid:
            masks = tp.sigmoid(masks)
        masks = self.mask_downsampler(masks)

        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)

        pos = tp.cast(self.position_encoding(x), x.dtype)

        return x, pos
