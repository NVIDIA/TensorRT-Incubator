import nvtripy as tp
import numpy as np

class TripyLayerNormEmbeddingPostprocessor:
    def __init__(self, embedding_dim, eps=1e-6):
        self._embedding_dim = embedding_dim
        self._eps = eps
        self._layernorm = tp.LayerNorm(embedding_dim, eps=eps)
        self._layernorm.initialize_dummy_parameters()
        # Set weights to all ones and bias to all zeros by default
        self._layernorm.weight = tp.Tensor(np.ones((embedding_dim,), dtype=np.float32))
        self._layernorm.bias = tp.Tensor(np.zeros((embedding_dim,), dtype=np.float32))

    def debug_str(self):
        return "ln"

    def forward(self, output_embeddings):
        output_embeddings = output_embeddings[..., :self._embedding_dim]
        return self._layernorm(output_embeddings)
