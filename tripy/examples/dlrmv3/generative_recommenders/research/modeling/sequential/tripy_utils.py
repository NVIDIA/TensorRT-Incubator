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
import nvtripy as tp


def tripy_get_current_embeddings(lengths: tp.Tensor, encoded_embeddings: tp.Tensor) -> tp.Tensor:
    """
    Args:
        lengths: (B,) x int
        encoded_embeddings: (B, N, D,) x float
    Returns:
        (B, D,) x float, where [i, :] == encoded_embeddings[i, lengths[i] - 1, :]
    """
    B, N, D = encoded_embeddings.shape
    flattened_offsets = (lengths - tp.Tensor(1)) + tp.arange(B, dtype=lengths.dtype) * N
    flat = tp.reshape(encoded_embeddings, (B * N, D))
    selected = tp.gather(flat, 0, flattened_offsets)
    return tp.reshape(selected, (B, D))
