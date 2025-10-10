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
import numpy as np
from typing import Tuple, Dict


class TripyInputFeaturesPreprocessorModule(tp.Module):
    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def forward(
        self,
        past_lengths: tp.Tensor,
        past_ids: tp.Tensor,
        past_embeddings: tp.Tensor,
        past_payloads: Dict[str, tp.Tensor],
    ) -> Tuple[tp.Tensor, tp.Tensor, tp.Tensor]:
        pass


# TODO: simplify the re-write with tensor methods
class TripyLearnablePositionalEmbeddingInputFeaturesPreprocessor(TripyInputFeaturesPreprocessorModule):
    def __init__(self, max_sequence_len, embedding_dim, dropout_rate):
        self._embedding_dim = embedding_dim
        self._pos_emb = tp.Embedding(max_sequence_len, embedding_dim)
        self._dropout_rate = dropout_rate
        self._emb_dropout = lambda x: x  # No-op, Tripy does not have Dropout
        self.reset_state()

    def debug_str(self):
        return f"posi_d{self._dropout_rate}"

    def reset_state(self):
        shape = self._pos_emb.weight.shape
        np_weight = np.random.normal(loc=0.0, scale=np.sqrt(1.0 / self._embedding_dim), size=shape)
        self._pos_emb.weight = tp.Tensor(np_weight.astype(np.float32))

    def forward(self, past_lengths, past_ids, past_embeddings, past_payloads):
        B, N = past_ids.shape
        D = past_embeddings.shape[-1]
        pos_idx = tp.unsqueeze(tp.arange(N), 0)
        pos_idx = tp.repeat(pos_idx, B, dim=0)
        pos_idx = tp.cast(pos_idx, tp.int32)
        user_embeddings = past_embeddings * (self._embedding_dim**0.5) + self._pos_emb(pos_idx)
        user_embeddings = self._emb_dropout(user_embeddings)
        valid_mask = tp.unsqueeze((past_ids != tp.Tensor(0).cast(past_ids.dtype)), -1)
        valid_mask = tp.cast(valid_mask, tp.float32)
        user_embeddings = user_embeddings * valid_mask
        return past_lengths, user_embeddings, valid_mask


class TripyLearnablePositionalEmbeddingRatedInputFeaturesPreprocessor(TripyInputFeaturesPreprocessorModule):
    def __init__(self, max_sequence_len, item_embedding_dim, dropout_rate, rating_embedding_dim, num_ratings):
        self._embedding_dim = item_embedding_dim + rating_embedding_dim
        self._pos_emb = tp.Embedding(max_sequence_len, self._embedding_dim)
        self._dropout_rate = dropout_rate
        self._emb_dropout = lambda x: x  # No-op for Dropout
        self._rating_emb = tp.Embedding(num_ratings, rating_embedding_dim)
        self.reset_state()

    def debug_str(self):
        return f"posir_d{self._dropout_rate}"

    def reset_state(self):
        shape = self._pos_emb.weight.shape
        np_weight = np.random.normal(loc=0.0, scale=np.sqrt(1.0 / self._embedding_dim), size=shape)
        self._pos_emb.weight = tp.Tensor(np_weight.astype(np.float32))
        shape_r = self._rating_emb.weight.shape
        np_weight_r = np.random.normal(loc=0.0, scale=np.sqrt(1.0 / self._embedding_dim), size=shape_r)
        self._rating_emb.weight = tp.Tensor(np_weight_r.astype(np.float32))

    def forward(self, past_lengths, past_ids, past_embeddings, past_payloads):
        B, N = past_ids.shape
        ratings = past_payloads["ratings"]
        user_embeddings = tp.concatenate([past_embeddings, self._rating_emb(tp.cast(ratings, tp.int32))], dim=-1)
        user_embeddings = user_embeddings * (self._embedding_dim**0.5)
        pos_idx = tp.unsqueeze(tp.arange(N), 0)
        pos_idx = tp.repeat(pos_idx, B, dim=0)
        pos_idx = tp.cast(pos_idx, tp.int32)
        user_embeddings = user_embeddings + self._pos_emb(pos_idx)
        user_embeddings = self._emb_dropout(user_embeddings)
        valid_mask = tp.unsqueeze((past_ids != tp.Tensor(0).cast(past_ids.dtype)), -1)
        valid_mask = tp.cast(valid_mask, tp.float32)
        user_embeddings = user_embeddings * valid_mask
        return past_lengths, user_embeddings, valid_mask


class TripyCombinedItemAndRatingInputFeaturesPreprocessor(TripyInputFeaturesPreprocessorModule):
    def __init__(self, max_sequence_len, item_embedding_dim, dropout_rate, num_ratings):
        self._embedding_dim = item_embedding_dim
        self._pos_emb = tp.Embedding(max_sequence_len * 2, item_embedding_dim)
        self._dropout_rate = dropout_rate
        self._emb_dropout = lambda x: x
        self._rating_emb = tp.Embedding(num_ratings, item_embedding_dim)
        self.reset_state()

    def debug_str(self):
        return f"combir_d{self._dropout_rate}"

    def reset_state(self):
        shape = self._pos_emb.weight.shape
        np_weight = np.random.normal(loc=0.0, scale=np.sqrt(1.0 / self._embedding_dim), size=shape)
        self._pos_emb.weight = tp.Tensor(np_weight.astype(np.float32))
        shape_r = self._rating_emb.weight.shape
        np_weight_r = np.random.normal(loc=0.0, scale=np.sqrt(1.0 / self._embedding_dim), size=shape_r)
        self._rating_emb.weight = tp.Tensor(np_weight_r.astype(np.float32))

    def get_preprocessed_ids(self, past_lengths, past_ids, past_embeddings, past_payloads):
        B, N = past_ids.shape
        return tp.reshape(
            tp.concatenate(
                [tp.unsqueeze(past_ids, 2), tp.unsqueeze(tp.cast(past_payloads["ratings"], past_ids.dtype), 2)], dim=2
            ),
            (B, N * 2),
        )

    def get_preprocessed_masks(self, past_lengths, past_ids, past_embeddings, past_payloads):
        B, N = past_ids.shape
        return tp.reshape(
            tp.expand(tp.unsqueeze((past_ids != tp.Tensor(0).cast(past_ids.dtype)), 2), (-1, -1, 2)), (B, N * 2)
        )

    def forward(self, past_lengths, past_ids, past_embeddings, past_payloads):
        B, N = past_ids.shape
        D = past_embeddings.shape[-1]
        user_embeddings = tp.concatenate(
            [past_embeddings, self._rating_emb(tp.cast(past_payloads["ratings"], tp.int32))], dim=2
        ) * (self._embedding_dim**0.5)
        user_embeddings = tp.reshape(user_embeddings, (B, N * 2, D))
        pos_idx = tp.unsqueeze(tp.arange(N * 2), 0)
        pos_idx = tp.repeat(pos_idx, B, dim=0)
        pos_idx = tp.cast(pos_idx, tp.int32)
        user_embeddings = user_embeddings + self._pos_emb(pos_idx)
        user_embeddings = self._emb_dropout(user_embeddings)
        valid_mask = tp.unsqueeze(
            self.get_preprocessed_masks(past_lengths, past_ids, past_embeddings, past_payloads), 2
        )
        valid_mask = tp.cast(valid_mask, tp.float32)
        user_embeddings = user_embeddings * valid_mask
        return past_lengths * 2, user_embeddings, valid_mask
