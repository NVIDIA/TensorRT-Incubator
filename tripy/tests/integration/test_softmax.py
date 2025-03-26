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
import torch


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        ((2, 2), 0),
        ((2,), 0),
        ((2, 2), None),
        ((2, 2), -1),
    ],
)
def test_softmax(input_shape, dim):
    input = tp.iota(input_shape, dtype=tp.float32)
    output = tp.softmax(input, dim=dim)

    torch_input = torch.from_dlpack(input)
    if dim is None:
        torch_input = torch_input.flatten()
    expected = torch.softmax(torch_input, dim if dim is not None else 0)
    expected = expected.reshape(input_shape)

    assert output.shape == input_shape
    assert tp.allclose(output, tp.Tensor(expected), atol=1e-4)
