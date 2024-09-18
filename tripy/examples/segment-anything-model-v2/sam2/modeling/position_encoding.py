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

from sam2.modeling.sam2_utils import cartesian_via_polar, mul_as_complex


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


def init_t_xy(end_x: tp.ShapeScalar, end_y: tp.ShapeScalar):
    t = tp.arange(end_x * end_y, dtype=tp.float32)
    if isinstance(end_x, tp.ShapeScalar) and isinstance(end_y, tp.ShapeScalar):
        end_x, end_y = tp.cast(end_x, tp.float32), tp.cast(end_y, tp.float32)
    t_x = t % end_x
    t_y = t // end_x
    return t_x, t_y


def compute_axial_cis(dim: int, end_x: tp.ShapeScalar, end_y: tp.ShapeScalar, theta: float = 10000.0):
    freqs_x = 1.0 / (theta ** (tp.cast(tp.arange(0, dim, 4)[: (dim // 4)], tp.float32) / dim))
    freqs_y = 1.0 / (theta ** (tp.cast(tp.arange(0, dim, 4)[: (dim // 4)], tp.float32) / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = tp.outer(t_x, freqs_x)
    freqs_y = tp.outer(t_y, freqs_y)
    freqs_cis_x = cartesian_via_polar(tp.ones_like(freqs_x), freqs_x)
    freqs_cis_y = cartesian_via_polar(tp.ones_like(freqs_y), freqs_y)
    return tp.concatenate([freqs_cis_x, freqs_cis_y], dim=-2)


def reshape_for_broadcast(freqs_cis: tp.Tensor, x: tp.Tensor):
    ndim = x.rank
    assert 0 <= 2 < ndim
    assert freqs_cis.shape == (x.shape[-3], x.shape[-2], 2)
    shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]
    return tp.reshape(freqs_cis, shape)


def apply_rotary_enc(
    xq: tp.Tensor,
    xk: tp.Tensor,
    freqs_cis: tp.Tensor,
    repeat_freqs_k: bool = False,
):
    xq_ = tp.reshape(tp.cast(xq, tp.float32), (*xq.shape[:-1], -1, 2))
    print(f"XQ Before: {xq.shape}, After: {xq_.shape}")
    xk_ = tp.reshape(tp.cast(xk, tp.float32), (*xk.shape[:-1], -1, 2)) if xk.shape[-2] != 0 else None

    print(f"XK Before: {xk.shape}, {xk_.shape}")
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = tp.flatten(mul_as_complex(xq_, freqs_cis), 3)
    if xk_ is None:
        # no keys to rotate, due to dropout
        return tp.cast(xq_out, xq.dtype), xk
    # repeat freqs along seq_len dim to match k seq_len
    if repeat_freqs_k:
        r = xk_.shape[-3] // xq_.shape[-3]
        if freqs_cis.is_cuda:
            freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 3)), r, 1)
        else:
            # tp.repeat on complex numbers may not be supported on non-CUDA devices
            # (freqs_cis has 5 dims and we repeat on dim 2) so we use expand + flatten
            freqs_cis = freqs_cis.unsqueeze(2).expand(-1, -1, r, -1, -1).flatten(2, 3)
    xk_out = tp.flatten(mul_as_complex(xq_, freqs_cis), 3)
    return tp.cast(xq_out, xq.dtype), tp.cast(xk_out, xk.dtype)
