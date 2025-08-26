# Test equivalence of PyTorch and Tripy RelativePositionalBias
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import torch
from sequential.hstu import RelativePositionalBias as TorchRelativePositionalBias
from sequential.tripy_hstu.relative_attention_bias_module import RelativePositionalBias as TripyRelativePositionalBias
import nvtripy as tp

def test_relative_positional_bias_equivalence():
    max_seq_len = 8
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # PyTorch version
    torch_mod = TorchRelativePositionalBias(max_seq_len)
    # Tripy version
    tripy_mod = TripyRelativePositionalBias(max_seq_len)
    # Copy weights from torch to tripy
    tripy_mod._w = tp.Tensor(torch_mod._w.detach().cpu().numpy())

    # all_timestamps is ignored, but must be correct shape
    dummy_timestamps = torch.zeros((1, max_seq_len), dtype=torch.int64)
    torch_out = torch_mod(dummy_timestamps)
    tripy_timestamps = tp.zeros((1, max_seq_len), dtype=tp.int64)
    tripy_out = tripy_mod.forward(tripy_timestamps)

    # Compare outputs
    torch_out_np = torch_out.detach().cpu().numpy()
    tripy_out_np = tripy_out.tolist()
    assert np.allclose(torch_out_np, tripy_out_np, atol=1e-5), f"Mismatch: {torch_out_np} vs {tripy_out_np}"
    print("RelativePositionalBias Tripy/PyTorch equivalence test passed!")

if __name__ == "__main__":
    test_relative_positional_bias_equivalence()
