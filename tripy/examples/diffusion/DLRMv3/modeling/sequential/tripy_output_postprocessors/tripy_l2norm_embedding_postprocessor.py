import nvtripy as tp

class TripyL2NormEmbeddingPostprocessor:
    def __init__(self, embedding_dim, eps=1e-6):
        self._embedding_dim = embedding_dim
        self._eps = eps

    def debug_str(self):
        return "l2"

    def forward(self, output_embeddings):
        output_embeddings = output_embeddings[..., :self._embedding_dim]
        norm = tp.sqrt(tp.sum(output_embeddings * output_embeddings, dim=-1, keepdim=True))
        norm = tp.maximum(norm, tp.Tensor(self._eps))
        return output_embeddings / norm
