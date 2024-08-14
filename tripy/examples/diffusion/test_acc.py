import tripy as tp 
import torch

# TODO - add e2e accuracy test

def check_equal(tp_array, torch_tensor, dtype=torch.float32, debug=False):
    if debug:
        a = torch.from_dlpack(tp_array).to(dtype)
        b = torch_tensor.to(dtype)
        diff = a - b
        print(f"tripy output shape: {a.shape}, torch output shape: {b.shape}")

        max_abs_diff = torch.max(torch.abs(diff))
        print(f"Maximum absolute difference: {max_abs_diff}\n")

        # Add small epsilon to denominator to avoid division by 0
        eps = 1e-8
        rel_diff = torch.abs(diff) / (torch.abs(b) + eps)
        max_rel_diff = torch.max(rel_diff)
        print(f"Maximum relative difference: {max_rel_diff}\n")
        
    assert torch.allclose(torch.from_dlpack(tp_array).to(dtype), torch_tensor.to(dtype)), f"\nTP Array:\n {tp_array} \n!= Torch Tensor:\n {torch_tensor}"