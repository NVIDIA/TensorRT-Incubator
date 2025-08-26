# Tripy translation of embedding_modules.py
import nvtripy as tp
import numpy as np

class TripyEmbeddingModule:
    def debug_str(self):
        raise NotImplementedError()
    def get_item_embeddings(self, item_ids):
        raise NotImplementedError()
    @property
    def item_embedding_dim(self):
        raise NotImplementedError()

class TripyLocalEmbeddingModule(TripyEmbeddingModule):
    def __init__(self, num_items, item_embedding_dim):
        self._item_embedding_dim = item_embedding_dim
        self._item_emb = tp.Embedding(num_items + 1, item_embedding_dim)
        self.reset_params()
    def debug_str(self):
        return f"local_emb_d{self._item_embedding_dim}"
    def reset_params(self):
        # Truncated normal init
        shape = self._item_emb.weight.shape
        np_weight = np.random.normal(loc=0.0, scale=0.02, size=shape)
        np_weight = np.clip(np_weight, -2*0.02, 2*0.02)
        self._item_emb.weight = tp.Tensor(np_weight.astype(np.float32))
    def get_item_embeddings(self, item_ids):
        return self._item_emb(item_ids)
    @property
    def item_embedding_dim(self):
        return self._item_embedding_dim

class TripyCategoricalEmbeddingModule(TripyEmbeddingModule):
    def __init__(self, num_items, item_embedding_dim, item_id_to_category_id):
        self._item_embedding_dim = item_embedding_dim
        self._item_emb = tp.Embedding(num_items + 1, item_embedding_dim)
        self._item_id_to_category_id = item_id_to_category_id
        self.reset_params()
    def debug_str(self):
        return f"cat_emb_d{self._item_embedding_dim}"
    def reset_params(self):
        shape = self._item_emb.weight.shape
        np_weight = np.random.normal(loc=0.0, scale=0.02, size=shape)
        np_weight = np.clip(np_weight, -2*0.02, 2*0.02)
        self._item_emb.weight = tp.Tensor(np_weight.astype(np.float32))
    def get_item_embeddings(self, item_ids):
        # item_ids = self._item_id_to_category_id[(item_ids - 1).clamp(min=0)] + 1
        idx = tp.maximum(item_ids - 1, tp.Tensor(0))
        idx = self._item_id_to_category_id[idx] + 1
        return self._item_emb(idx)
    @property
    def item_embedding_dim(self):
        return self._item_embedding_dim
