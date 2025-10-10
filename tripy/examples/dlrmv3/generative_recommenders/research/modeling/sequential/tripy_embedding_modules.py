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

import nvtripy as tp

from generative_recommenders.research.modeling.tripy_initialization import tripy_truncated_normal


class TripyEmbeddingModule(tp.Module):
    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def get_item_embeddings(self, item_ids: tp.Tensor) -> tp.Tensor:
        pass

    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass


class TripyLocalEmbeddingModule(TripyEmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb = tp.Embedding(num_items + 1, item_embedding_dim)
        self.reset_params()

    def debug_str(self) -> str:
        return f"local_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(f"Initialize {name} as truncated normal: {params.data.size()} params")
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: tp.Tensor) -> tp.Tensor:
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class TripyCategoricalEmbeddingModule(TripyEmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        item_id_to_category_id: tp.Tensor,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb: tp.Embedding = tp.Embedding(num_items + 1, item_embedding_dim)
        self.register_buffer("_item_id_to_category_id", item_id_to_category_id)
        self.reset_params()

    def debug_str(self) -> str:
        return f"cat_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(f"Initialize {name} as truncated normal: {params.data.size()} params")
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: tp.Tensor) -> tp.Tensor:
        idx = tp.maximum(item_ids - 1, tp.Tensor(0))
        item_ids = self._item_id_to_category_id[idx] + 1
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim
