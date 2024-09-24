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
import numpy as np
import tripy as tp
from typing import Optional, Tuple


class PositionEmbeddingRandom(tp.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.positional_encoding_gaussian_matrix = tp.Parameter(
            tp.Tensor(
                scale * np.random.randn(2, num_pos_feats).astype(np.float32),
                dtype=tp.float32,
            )
        )

    def _pe_encoding(self, coords: tp.Tensor) -> tp.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return tp.concatenate([tp.sin(coords), tp.cos(coords)], dim=-1)

    def __call__(self, size: Tuple[int, int]) -> tp.Tensor:
        return self.forward(size)

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

    def forward_with_coords(
        self, coords_input: tp.Tensor, image_size: Tuple[int, int]
    ) -> tp.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        new_x_coords = coords_input[:, :, 0] / image_size[1]
        new_y_coords = coords_input[:, :, 1] / image_size[0]

        # Combine the updated x and y coordinates into a new tensor
        new_coords = tp.stack([new_x_coords, new_y_coords], dim=-1)
        return self._pe_encoding(tp.cast(new_coords, tp.float32))  # B x N x C
