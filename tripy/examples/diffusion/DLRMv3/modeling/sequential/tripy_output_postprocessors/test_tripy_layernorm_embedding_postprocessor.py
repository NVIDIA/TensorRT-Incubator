import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import torch
import nvtripy as tp
from sequential.output_postprocessors import LayerNormEmbeddingPostprocessor
from sequential.tripy_output_postprocessors.tripy_layernorm_embedding_postprocessor import TripyLayerNormEmbeddingPostprocessor

def torch_to_tripy_tensor(tensor):
    arr = tensor.detach().cpu().numpy()
    if tensor.dtype in [torch.int32, torch.int64]:
        return tp.Tensor(arr.astype(np.int32))
    return tp.Tensor(arr.astype(np.float32))

def test_layernorm_embedding_equivalence():
    B, D = 3, 8
    torch_mod = LayerNormEmbeddingPostprocessor(D)
    tripy_mod = TripyLayerNormEmbeddingPostprocessor(D)
    output_embeddings = torch.randn(B, D)
    torch_out = torch_mod(output_embeddings)
    tripy_out = tripy_mod.forward(torch_to_tripy_tensor(output_embeddings))
    np.testing.assert_allclose(torch_out.detach().cpu().numpy(), tripy_out.tolist(), rtol=1e-5, atol=1e-6)
