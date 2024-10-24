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
from sam2.modeling.tripy.sam2_utils import cartesian_via_polar, mul_as_complex


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


def init_t_xy(end_x: tp.ShapeScalar, end_y: tp.ShapeScalar):
    t = tp.arange(end_x * end_y, dtype=tp.float32)
    if isinstance(end_x, tp.ShapeScalar) and isinstance(end_y, tp.ShapeScalar):
        end_x, end_y = tp.cast(end_x, tp.float32), tp.cast(end_y, tp.float32)
    t_x = t % end_x
    t_y = t // end_x
    return t_x, t_y


def compute_axial_cis(
    dim: int, end_x: tp.ShapeScalar, end_y: tp.ShapeScalar, theta: float = 10000.0
):
    freqs_x = 1.0 / (
        theta ** (tp.cast(tp.arange(0, dim, 4)[: (dim // 4)], tp.float32) / dim)
    )
    freqs_y = 1.0 / (
        theta ** (tp.cast(tp.arange(0, dim, 4)[: (dim // 4)], tp.float32) / dim)
    )

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = tp.outer(t_x, freqs_x)
    freqs_y = tp.outer(t_y, freqs_y)
    freqs_cis_x = cartesian_via_polar(tp.ones_like(freqs_x), freqs_x)
    freqs_cis_y = cartesian_via_polar(tp.ones_like(freqs_y), freqs_y)
    return tp.concatenate([freqs_cis_x, freqs_cis_y], dim=-2)


def reshape_for_broadcast(freqs_cis: tp.Tensor, x: tp.Tensor):
    ndim = x.rank
    shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]
    return tp.reshape(freqs_cis, shape)


def apply_rotary_enc(
    xq: tp.Tensor,
    xk: tp.Tensor,
    freqs_cis: tp.Tensor,
    repeat_freqs_k: bool = False,
):
    xq_ = tp.reshape(xq, (*xq.shape[:-1], -1, 2))
    xk_ = tp.reshape(xk, (*xk.shape[:-1], -1, 2))

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = mul_as_complex(xq_, freqs_cis)
    xq_out = tp.reshape(xq_out, (xq_out.shape[0], xq_out.shape[1], xq_out.shape[2], -1))

    # repeat freqs along seq_len dim to match k seq_len
    if repeat_freqs_k:
        r = xk_.shape[-3] // xq_.shape[-3]
        freqs_cis = tp.flatten(
            tp.expand(tp.unsqueeze(freqs_cis, 2), (-1, -1, r, -1, -1, -1)), 2, 3
        )

    xk_out = tp.flatten(mul_as_complex(xk_, freqs_cis), 3)
    return tp.cast(xq_out, xq.dtype), tp.cast(xk_out, xk.dtype)
