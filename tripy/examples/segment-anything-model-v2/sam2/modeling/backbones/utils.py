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

from typing import Tuple

import tripy as tp


def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.
    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape
    # padding is not triggered
    Hp, Wp = H, W
    x = tp.reshape(x, (B, Hp // window_size, window_size, Wp // window_size, window_size, C))
    x = tp.permute(x, (0, 1, 3, 2, 4, 5))
    windows = tp.reshape(x, (-1, window_size, window_size, C))
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.
    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)

    x = tp.reshape(windows, (B, Hp // window_size, Wp // window_size, window_size, window_size, -1))
    x = tp.permute(x, (0, 1, 3, 2, 4, 5))  # [B, Hp//window_size, window_size, Wp//window_size, window_size, C]
    x = tp.reshape(x, (B, Hp, Wp, -1))  # [B, Hp, Wp, C]
    return x


class PatchEmbed(tp.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, ...] = (7, 7),
        stride: Tuple[int, ...] = (4, 4),
        padding: Tuple[int, ...] = (3, 3),
        in_chans: int = 3,
        embed_dim: int = 768,
        dtype: tp.dtype = tp.float32,
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        padding = ((padding[0], padding[0]), (padding[1], padding[1]))
        self.proj = tp.Conv(
            in_chans,
            embed_dim,
            kernel_dims=kernel_size,
            stride=stride,
            padding=padding,
            dtype=dtype,
        )

    def __call__(self, x):
        x = self.proj(x)
        x = tp.permute(x, (0, 2, 3, 1))  # [B, C, H, W] -> [B, H, W, C]
        return x
