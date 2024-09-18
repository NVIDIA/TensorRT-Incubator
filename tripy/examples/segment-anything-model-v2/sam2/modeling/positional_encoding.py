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
from typing import Optional, Tuple

import tripy as tp
import numpy as np


class PositionEmbeddingSine(tp.Module):
    def __init__(
        self,
        num_pos_feats,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: tp.Tensor):
        batch, seq_length, hidden_dim = x.shape
        y_embed = tp.reshape(tp.arange(1, seq_length + 1, 1), (1, seq_length, 1)) * tp.ones(
            (batch, seq_length, hidden_dim)
        )
        x_embed = tp.reshape(tp.arange(1, hidden_dim + 1, 1), (1, 1, hidden_dim)) * tp.ones(
            (batch, seq_length, hidden_dim)
        )

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = tp.arange(self.num_pos_feats, dtype=tp.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = tp.unsqueeze(x_embed, 3)[:, :, :] / dim_t
        pos_y = tp.unsqueeze(y_embed, 3)[:, :, :] / dim_t
        pos_x = tp.flatten(tp.stack([tp.sin(pos_x[:, :, :, 0::2]), tp.cos(pos_x[:, :, :, 1::2])], dim=4), 3)
        pos_y = tp.flatten(tp.stack([tp.sin(pos_y[:, :, :, 0::2]), tp.cos(pos_y[:, :, :, 1::2])], dim=4), 3)
        pos = tp.concatenate((pos_y, pos_x), dim=3)
        pos = tp.permute(pos, (0, 3, 1, 2))
        return pos


# p = PositionEmbeddingSine(1024)
# print(p.forward(tp.ones((1, 128, 128))))


class PositionEmbeddingRandom(tp.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.positional_encoding_gaussian_matrix = tp.Parameter(
            tp.Tensor(scale * np.random.randn(2, num_pos_feats).astype(np.float32), dtype=tp.float32)
        )

    def _pe_encoding(self, coords: tp.Tensor) -> tp.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return tp.concatenate([tp.sin(coords), tp.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> tp.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        grid = tp.ones((h, w), dtype=tp.float32)
        y_embed = tp.cumsum(grid, dim=0) - 0.5
        x_embed = tp.cumsum(grid, dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(tp.stack([x_embed, y_embed], dim=-1))
        return tp.permute(pe, (2, 0, 1))  # C x H x W

    def forward_with_coords(self, coords_input: tp.Tensor, image_size: Tuple[int, int]) -> tp.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        new_x_coords = coords_input[:, :, 0] / image_size[1]
        new_y_coords = coords_input[:, :, 1] / image_size[0]

        # Combine the updated x and y coordinates into a new tensor
        new_coords = tp.stack([new_x_coords, new_y_coords], dim=-1)
        return self._pe_encoding(tp.cast(new_coords, tp.float32))  # B x N x C


# p = PositionEmbeddingRandom()
# print(p.forward((10,20)))
# print(p.forward_with_coords(tp.ones((1,100,100)), (10,10)))
