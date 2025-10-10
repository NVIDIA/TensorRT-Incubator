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

import abc


class TripyOutputPostprocessorModule(tp.Module):
    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def forward(self, output_embeddings: tp.Tensor) -> tp.Tensor:
        pass


class TripyL2NormEmbeddingPostprocessor(TripyOutputPostprocessorModule):
    def __init__(self, embedding_dim, eps=1e-6):
        self._embedding_dim = embedding_dim
        self._eps = eps

    def debug_str(self) -> str:
        return "l2"

    def forward(self, output_embeddings: tp.Tensor) -> tp.Tensor:
        output_embeddings = output_embeddings[..., : self._embedding_dim]
        norm = tp.sqrt(tp.sum(output_embeddings * output_embeddings, dim=-1, keepdim=True))
        norm = tp.maximum(norm, tp.Tensor(self._eps))
        return output_embeddings / norm


class TripyLayerNormEmbeddingPostprocessor(TripyOutputPostprocessorModule):
    def __init__(self, embedding_dim, eps=1e-6):
        self._embedding_dim = embedding_dim
        self._eps = eps
        self._layernorm = tp.LayerNorm(embedding_dim, eps=eps)
        self._layernorm.initialize_dummy_parameters()
        # Set weights to all ones and bias to all zeros by default
        self._layernorm.weight = tp.Tensor(np.ones((embedding_dim,), dtype=np.float32))
        self._layernorm.bias = tp.Tensor(np.zeros((embedding_dim,), dtype=np.float32))

    def debug_str(self) -> str:
        return "ln"

    def forward(self, output_embeddings: tp.Tensor) -> tp.Tensor:
        output_embeddings = output_embeddings[..., : self._embedding_dim]
        return self._layernorm(output_embeddings)
