# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import torch
import numpy as np
from argparse import Namespace
from skimage.metrics import structural_similarity

from examples.diffusion.example import tripy_diffusion, hf_diffusion


# Utility for debugging hidden states in model via floating-point comparison
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

    assert torch.allclose(
        torch.from_dlpack(tp_array).to(dtype), torch_tensor.to(dtype)
    ), f"\nTP Array:\n {tp_array} \n!= Torch Tensor:\n {torch_tensor}"


@pytest.mark.l1
class TestConvolution:
    def test_ssim(self):
        args = Namespace(
            steps=50,
            prompt="a beautiful photograph of Mt. Fuji during cherry blossom",
            out="temp.png",
            fp16=True,
            seed=420,
            guidance=7.5,
            torch_inference=False,
            hf_token="",
            engine_dir="diffusion_engines",
        )
        tp_img, _ = tripy_diffusion(args)
        tp_img = np.array(tp_img.convert("L"))

        torch_img, _ = hf_diffusion(args)
        torch_img = np.array(torch_img.convert("L"))

        ssim = structural_similarity(tp_img, torch_img)
        assert ssim >= 0.80, f"Structural Similarity score expected >= 0.80 but got {ssim}"
