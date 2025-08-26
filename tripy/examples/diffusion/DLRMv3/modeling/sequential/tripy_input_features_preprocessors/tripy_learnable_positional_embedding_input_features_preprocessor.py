import nvtripy as tp
import numpy as np
from .tripy_input_features_preprocessor_module import TripyInputFeaturesPreprocessorModule

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
        user_embeddings = past_embeddings * (self._embedding_dim ** 0.5) + self._pos_emb(pos_idx)
        user_embeddings = self._emb_dropout(user_embeddings)
        valid_mask = tp.unsqueeze((past_ids != tp.Tensor(0)), -1)
        valid_mask = tp.cast(valid_mask, tp.float32)
        user_embeddings = user_embeddings * valid_mask
        return past_lengths, user_embeddings, valid_mask
