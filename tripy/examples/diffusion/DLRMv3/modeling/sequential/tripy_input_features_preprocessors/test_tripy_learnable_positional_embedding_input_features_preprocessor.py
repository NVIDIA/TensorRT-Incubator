
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import torch
import nvtripy as tp
from sequential.input_features_preprocessors import LearnablePositionalEmbeddingInputFeaturesPreprocessor
from sequential.tripy_input_features_preprocessors.tripy_learnable_positional_embedding_input_features_preprocessor import TripyLearnablePositionalEmbeddingInputFeaturesPreprocessor

def torch_to_tripy_tensor(tensor):
    arr = tensor.detach().cpu().numpy()
    if tensor.dtype in [torch.int32, torch.int64]:
        return tp.Tensor(arr.astype(np.int32))
    return tp.Tensor(arr.astype(np.float32))

def test_learnable_positional_embedding_equivalence():
    B, N, D = 2, 5, 4
    torch_mod = LearnablePositionalEmbeddingInputFeaturesPreprocessor(N, D, 0.0)
    tripy_mod = TripyLearnablePositionalEmbeddingInputFeaturesPreprocessor(N, D, 0.0)
    # Copy weights
    tripy_mod._pos_emb.weight = tp.Tensor(torch_mod._pos_emb.weight.detach().cpu().numpy())
    past_lengths = torch.randint(1, N+1, (B,))
    past_ids = torch.randint(0, 10, (B, N))
    past_embeddings = torch.randn(B, N, D)
    past_payloads = {}
    torch_out = torch_mod(past_lengths, past_ids, past_embeddings, past_payloads)
    tripy_out = tripy_mod.forward(
        torch_to_tripy_tensor(past_lengths),
        torch_to_tripy_tensor(past_ids),
        torch_to_tripy_tensor(past_embeddings),
        past_payloads
    )
    np.testing.assert_allclose(torch_out[1].detach().cpu().numpy(), tripy_out[1].tolist(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(torch_out[2].detach().cpu().numpy(), tripy_out[2].tolist(), rtol=1e-5, atol=1e-6)
