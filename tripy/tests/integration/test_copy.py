# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import cupy as cp
import numpy as np
import nvtripy as tp

import pytest


class TestCopy:
    @pytest.mark.parametrize(
        "copy_func",
        [
            lambda tensor, device: tp.copy(tensor, device),  # Free function
            lambda tensor, device: tensor.copy(device),  # Tensor method
        ],
    )
    def test_copy_tensor_method(self, copy_func):
        """Test that both copy methods work with compilation."""
        gpu_tensor = tp.Tensor(cp.ones((2, 2), dtype=cp.float32))
        assert gpu_tensor.device.kind == "gpu"

        cpu_tensor = copy_func(gpu_tensor, tp.device("cpu"))

        assert cpu_tensor.device.kind == "cpu"
        # If the tensor is really in CPU memory, we should be able to construct a NumPy array from it
        assert np.from_dlpack(cpu_tensor).shape == (2, 2)

    @pytest.mark.parametrize(
        "copy_func",
        [
            lambda tensor, device: tp.copy(tensor, device),  # Free function
            lambda tensor, device: tensor.copy(device),  # Tensor method
        ],
    )
    def test_to_gpu(self, copy_func):
        cpu_tensor = tp.Tensor(np.ones((2, 2), dtype=np.float32))
        assert cpu_tensor.device.kind == "cpu"

        gpu_tensor = copy_func(cpu_tensor, tp.device("gpu"))
        assert gpu_tensor.device.kind == "gpu"

        # If the tensor is really in GPU memory, we should be able to construct a Cupy array from it
        assert cp.from_dlpack(gpu_tensor).shape == (2, 2)
