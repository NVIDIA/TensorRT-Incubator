# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");Add commentMore actions
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
from typing import Optional

import nvtripy as tp


def scaled_dot_product_attention(
    query: tp.Tensor,
    key: tp.Tensor,
    value: tp.Tensor,
    embedding_dim: int,
    attn_mask: Optional[tp.Tensor] = None,
) -> tp.Tensor:
    dtype = query.dtype
    if attn_mask is not None and attn_mask.dtype == tp.bool:
        attn_mask = tp.where((attn_mask == 0), tp.cast(tp.Tensor(-float("inf")), dtype=dtype), 0.0)
    if attn_mask is not None:
        attn_mask = tp.cast(attn_mask, dtype)
    k_t = tp.transpose(key, -2, -1)
    qk = (query @ k_t) * (1.0 / math.sqrt(embedding_dim))
    return tp.softmax((qk + attn_mask) if attn_mask is not None else qk, -1) @ value


def clamp(tensor: tp.Tensor, min: int, max: int):
    return tp.minimum(tp.maximum(tensor, min), max)
