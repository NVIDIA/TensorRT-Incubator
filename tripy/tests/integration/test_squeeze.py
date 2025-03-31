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
import nvtripy as tp
import pytest


class TestSqueeze:
    @pytest.mark.parametrize(
        "input_shape, dims, expected_shape",
        [
            ((1, 2, 1), 0, (2, 1)),  # Squeeze first dimension
            ((1, 2, 1), (0, 2), (2,)),  # Squeeze first and third dimensions
            ((1, 2, 1), tuple(), (1, 2, 1)),  # No dimensions to squeeze
            ((1, 2, 1), (-3, -1), (2,)),  # Squeeze using negative dimensions
        ],
    )
    def test_squeeze(self, input_shape, dims, expected_shape):
        input_tensor = tp.ones(input_shape, dtype=tp.float32)
        output_tensor = tp.squeeze(input_tensor, dims=dims)
        assert output_tensor.shape == expected_shape
