import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import torch
import nvtripy as tp

from similarity_module import SequentialEncoderWithLearnedSimilarityModule
from sequential.tripy_similarity_module.tripy_sequential_encoder_with_learned_similarity_module import TripySequentialEncoderWithLearnedSimilarityModule


# PyTorch dummy similarity module
class TorchDummySimilarityModule(torch.nn.Module):
    def forward(self, query_embeddings, item_embeddings, item_ids, **kwargs):
        # Just return the sum for testing
        return torch.sum(query_embeddings) + torch.sum(item_embeddings) + torch.sum(item_ids)

# Tripy dummy similarity module
class DummySimilarityModule:
    def __call__(self, query_embeddings, item_embeddings, item_ids, **kwargs):
        return tp.sum(query_embeddings) + tp.sum(item_embeddings) + tp.sum(item_ids)

def test_tripy_sequential_encoder_with_learned_similarity_module():
    # PyTorch version
    torch_ndp_module = TorchDummySimilarityModule()
    torch_mod = SequentialEncoderWithLearnedSimilarityModule(torch_ndp_module)
    torch_mod.get_item_embeddings = lambda item_ids: torch.ones((2, 3, 4))  # Dummy if needed
    torch_query_embeddings = torch.ones((2, 4))
    torch_item_ids = torch.ones((2, 3))
    torch_item_embeddings = torch.ones((2, 3, 4))
    torch_result = torch_mod.similarity_fn(torch_query_embeddings, torch_item_ids, torch_item_embeddings)

    # Tripy version
    ndp_module = DummySimilarityModule()
    tripy_mod = TripySequentialEncoderWithLearnedSimilarityModule(ndp_module)
    query_embeddings = tp.ones((2, 4))
    item_ids = tp.ones((2, 3))
    item_embeddings = tp.ones((2, 3, 4))
    tripy_result = tripy_mod.similarity_fn(query_embeddings, item_ids, item_embeddings)

    # Compare outputs
    assert np.allclose(tripy_result.tolist(), torch_result.detach().cpu().numpy())
