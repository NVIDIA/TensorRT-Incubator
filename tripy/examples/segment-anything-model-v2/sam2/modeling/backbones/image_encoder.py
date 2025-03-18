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


class ImageEncoder(tp.Module):

    def __init__(
        self,
        trunk: tp.Module,
        neck: tp.Module,
        scalp: int = 0,
    ):
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        assert (
            self.trunk.channel_list == self.neck.backbone_channel_list
        ), f"Channel dims of trunk and neck do not match. Trunk: {self.trunk.channel_list}, neck: {self.neck.backbone_channel_list}"
        self.compiled_executable = None

    def forward(self, x):
        # __call__ returns a dict, not tensors
        # thus we need to only compile this forward function
        # Forward through backbone
        return self.neck(self.trunk(x))

    def __call__(self, sample: tp.Tensor):
        import torch

        # Forward through backbone
        if self.compiled_executable:
            features_pos = self.compiled_executable(sample)
            tp.default_stream().synchronize()
        else:
            features_pos = self.forward(sample)
        for i in range(len(features_pos)):
            features_pos[i] = torch.from_dlpack(features_pos[i])
        n = len(self.neck.backbone_channel_list)
        features = list(features_pos[:n])
        pos = list(features_pos[n:])
        if self.scalp > 0:
            # Discard the lowest resolution features
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]
        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        return output


class FpnNeck(tp.Module):

    def __init__(
        self,
        position_encoding: tp.Module,  # TODO: replace this with shapes
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
        dtype: str = "float32",
    ):
        super().__init__()
        self.dtype = getattr(tp, dtype)
        self.convs = []
        self.backbone_channel_list = backbone_channel_list
        for dim in backbone_channel_list:
            make_2d_tuple = lambda x: 2 * (x,)
            self.convs.append(
                tp.Conv(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_dims=make_2d_tuple(kernel_size),
                    stride=make_2d_tuple(stride),
                    dtype=self.dtype,
                )
            )
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type

        if fpn_top_down_levels is None:
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

        # position embedding only depends on input shape
        # so we generate static embeddings ahead of time
        self.position_encoding = []
        position_encoding_shapes = [[256, 256], [128, 128], [64, 64], [32, 32]]
        for s in position_encoding_shapes:
            self.position_encoding.append(position_encoding.generate_static_embedding([1, 256] + s, dtype=dtype))

    def __call__(self, xs: List[tp.Tensor]):

        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        # fpn forward pass
        prev_features = None
        # forward in top-down order (from low to high resolution)
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = tp.resize(
                    tp.cast(prev_features, self.dtype),
                    output_shape=(prev_features.shape[0], 256, 64, 64),
                    mode=self.fpn_interp_model,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = tp.cast(self.position_encoding[i], x_out.dtype)

        return *out, *pos
