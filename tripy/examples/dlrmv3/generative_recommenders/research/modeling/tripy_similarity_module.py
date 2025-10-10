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
import abc
from typing import Dict, Tuple, Optional

import nvtripy as tp


class TripySimilarityModule(tp.Module):
    """
    Interface enabling interfacing with various similarity functions.

    While the discussions in our initial ICML'24 paper are based on inner products
    for simplicity, we provide this interface (SimilarityModule) to support various
    learned similarities at the retrieval stage, such as MLPs, Factorization Machines
    (FMs), and Mixture-of-Logits (MoL), which we discussed in
    - Revisiting Neural Retrieval on Accelerators (KDD'23), and
    - Retrieval with Learned Similarities (https://arxiv.org/abs/2407.15462).
    """

    @abc.abstractmethod
    def forward(
        self,
        query_embeddings: tp.Tensor,
        item_embeddings: tp.Tensor,
        **kwargs,
    ) -> Tuple[tp.Tensor, Dict[str, tp.Tensor]]:
        """
        Args:
            query_embeddings: (B, input_embedding_dim) x tp.float.
            item_embeddings: (1/B, X, item_embedding_dim) x float.
            **kwargs: Implementation-specific keys/values (e.g.,
                item ids / sideinfo, etc.)

        Returns:
            A tuple of (
                (B, X,) similarity values,
                keyed outputs representing auxiliary losses at training time.
            ).
        """
        pass


class TripyDotProductSimilarity(TripySimilarityModule):
    """Tripy implementation of dot product similarity, matching PyTorch DotProductSimilarity."""

    def __init__(self) -> None:
        super().__init__()

    def debug_str(self) -> str:
        return "dp"

    def forward(
        self,
        query_embeddings: tp.Tensor,
        item_embeddings: tp.Tensor,
        **kwargs,
    ) -> Tuple[tp.Tensor, Dict[str, tp.Tensor]]:
        """
        Args:
            query_embeddings: (B, D) or (B * r, D) x float.
            item_embeddings: (1, X, D) or (B, X, D) x float.

        Returns:
            (B, X) x float similarities and empty debug dict.
        """
        B_I, X, D = item_embeddings.shape

        if B_I == 1:
            # [B, D] x ([1, X, D] -> [D, X]) => [B, X]
            item_t = tp.transpose(tp.squeeze(item_embeddings, 0), (1, 0))  # [D, X]
            similarities = tp.matmul(query_embeddings, item_t)  # [B, X]
            return similarities, {}

        elif query_embeddings.shape[0] != B_I:
            # (B * r, D) x (B, X, D) case
            B = B_I
            r = query_embeddings.shape[0] // B
            query_reshaped = tp.reshape(query_embeddings, (B, r, D))  # [B, r, D]
            item_transposed = tp.transpose(item_embeddings, (0, 2, 1))  # [B, D, X]
            similarities_3d = tp.matmul(query_reshaped, item_transposed)  # [B, r, X]
            similarities = tp.reshape(similarities_3d, (-1, X))  # [B*r, X]
            return similarities, {}

        else:
            # [B, X, D] x ([B, D] -> [B, D, 1]) => [B, X, 1] -> [B, X]
            query_expanded = tp.unsqueeze(query_embeddings, 2)  # [B, D, 1]
            similarities_3d = tp.matmul(item_embeddings, query_expanded)  # [B, X, 1]
            similarities = tp.squeeze(similarities_3d, 2)  # [B, X]
            return similarities, {}


class TripySequentialEncoderWithLearnedSimilarityModule(tp.Module):
    def __init__(self, ndp_module: TripySimilarityModule) -> None:
        super().__init__()

        self._ndp_module: TripySimilarityModule = ndp_module

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    def similarity_fn(
        self, query_embeddings: tp.Tensor, item_ids: tp.Tensor, item_embeddings: Optional[tp.Tensor] = None, **kwargs
    ) -> tp.Tensor:
        assert len(query_embeddings.shape) == 2, "len(query_embeddings.shape) must be 2"
        assert len(item_ids.shape) == 2, "len(item_ids.shape) must be 2"
        if item_embeddings is None:
            item_embeddings = self.get_item_embeddings(item_ids)
        assert len(item_embeddings.shape) == 3, "len(item_embeddings.shape) must be 3"
        return self._ndp_module(
            query_embeddings=query_embeddings,  # (B, query_embedding_dim)
            item_embeddings=item_embeddings,  # (1/B, X, item_embedding_dim)
            item_ids=item_ids,
            **kwargs,
        )
