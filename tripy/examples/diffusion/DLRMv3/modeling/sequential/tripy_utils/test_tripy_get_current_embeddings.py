import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import torch
import nvtripy as tp
from sequential.utils import get_current_embeddings
from sequential.tripy_utils.tripy_get_current_embeddings import tripy_get_current_embeddings

def torch_to_tripy_tensor(tensor):
    arr = tensor.detach().cpu().numpy()
    if tensor.dtype in [torch.int32, torch.int64]:
        return tp.Tensor(arr.astype(np.int32))
    return tp.Tensor(arr.astype(np.float32))

def test_tripy_get_current_embeddings_equivalence():
    B, N, D = 4, 6, 3
    lengths = torch.randint(1, N+1, (B,))
    encoded_embeddings = torch.randn(B, N, D)
    torch_out = get_current_embeddings(lengths, encoded_embeddings)
    tripy_out = tripy_get_current_embeddings(torch_to_tripy_tensor(lengths), torch_to_tripy_tensor(encoded_embeddings))
    np.testing.assert_allclose(torch_out.detach().cpu().numpy(), tripy_out.tolist(), rtol=1e-5, atol=1e-6)
