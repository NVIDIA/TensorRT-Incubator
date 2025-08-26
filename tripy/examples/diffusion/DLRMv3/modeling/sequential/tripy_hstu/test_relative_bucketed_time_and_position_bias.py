# Test equivalence of PyTorch and Tripy RelativeBucketedTimeAndPositionBasedBias
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import torch
from sequential.hstu import RelativeBucketedTimeAndPositionBasedBias as TorchRelBias
from sequential.tripy_hstu.relative_bucketed_time_and_position_bias import RelativeBucketedTimeAndPositionBasedBias as TripyRelBias
import nvtripy as tp

def bucketization_fn_torch(x):
    return (torch.log(torch.abs(x).clamp(min=1)) / 0.301).long()

def bucketization_fn_tripy(x):
    # Cast to float32 before log, then cast result to int64
    x_float = abs(x).cast(tp.float32)
    return (tp.log(tp.maximum(x_float, 1.0)) / 0.301).cast(tp.int64)

def test_relative_bucketed_time_and_position_bias_equivalence():
    max_seq_len = 8
    num_buckets = 5
    torch.manual_seed(42)
    np.random.seed(42)
    # PyTorch version
    torch_mod = TorchRelBias(max_seq_len, num_buckets, bucketization_fn_torch)
    # Tripy version
    tripy_mod = TripyRelBias(max_seq_len, num_buckets, bucketization_fn_tripy)
    # Copy weights
    tripy_mod._ts_w = tp.Tensor(torch_mod._ts_w.detach().cpu().numpy())
    tripy_mod._pos_w = tp.Tensor(torch_mod._pos_w.detach().cpu().numpy())
    # Input
    B = 2
    all_timestamps = torch.randint(0, 100, (B, max_seq_len), dtype=torch.int64)
    torch_out = torch_mod(all_timestamps)
    tripy_in = tp.Tensor(all_timestamps.detach().cpu().numpy()).cast(tp.int64)
    tripy_out = tripy_mod.forward(tripy_in)
    # Compare
    torch_out_np = torch_out.detach().cpu().numpy()
    tripy_out_np = np.array(tripy_out.tolist())
    max_abs_diff = np.max(np.abs(torch_out_np - tripy_out_np))
    print(f"Max absolute difference: {max_abs_diff}")
    assert np.allclose(torch_out_np, tripy_out_np, atol=1e-5), f"Mismatch: {torch_out_np} vs {tripy_out_np}"
    print("RelativeBucketedTimeAndPositionBasedBias Tripy/PyTorch equivalence test passed!")

if __name__ == "__main__":
    test_relative_bucketed_time_and_position_bias_equivalence()
