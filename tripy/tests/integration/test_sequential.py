#
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
#

import torch

import tripy as tp


class TestSequential:

    def test_forward_pass_accuracy(self):
        # Initialize Sequential with two Linear layers in both PyTorch and Tripy
        torch_model = torch.nn.Sequential(
            torch.nn.Linear(1, 3, dtype=torch.float32), torch.nn.Linear(3, 2, dtype=torch.float32)
        )
        tp_model = tp.Sequential(tp.Linear(1, 3, dtype=tp.float32), tp.Linear(3, 2, dtype=tp.float32))

        tp_model[0].weight = tp.Parameter(torch_model[0].weight.detach())
        tp_model[0].bias = tp.Parameter(torch_model[0].bias.detach())
        tp_model[1].weight = tp.Parameter(torch_model[1].weight.detach())
        tp_model[1].bias = tp.Parameter(torch_model[1].bias.detach())

        input_tensor = torch.tensor([[1.0]], dtype=torch.float32)
        tp_input = tp.Tensor(input_tensor, dtype=tp.float32)

        tp_output = tp.copy(tp_model(tp_input), tp.device("cpu"))

        torch_model.eval()
        with torch.no_grad():
            torch_output = torch_model(input_tensor)

        rtol_ = 2e-7
        assert torch.allclose(torch.from_dlpack(tp_output), torch_output, rtol=rtol_)

    def test_nested_sequential_accuracy(self):
        torch_model = torch.nn.Sequential(
            torch.nn.Linear(1, 3, dtype=torch.float32),
            torch.nn.Sequential(torch.nn.Linear(3, 4, dtype=torch.float32), torch.nn.Linear(4, 2, dtype=torch.float32)),
        )
        tp_model = tp.Sequential(
            tp.Linear(1, 3, dtype=tp.float32),
            tp.Sequential(tp.Linear(3, 4, dtype=tp.float32), tp.Linear(4, 2, dtype=tp.float32)),
        )

        tp_model[0].weight = tp.Parameter(torch_model[0].weight.detach())
        tp_model[0].bias = tp.Parameter(torch_model[0].bias.detach())
        tp_model[1][0].weight = tp.Parameter(torch_model[1][0].weight.detach())
        tp_model[1][0].bias = tp.Parameter(torch_model[1][0].bias.detach())
        tp_model[1][1].weight = tp.Parameter(torch_model[1][1].weight.detach())
        tp_model[1][1].bias = tp.Parameter(torch_model[1][1].bias.detach())

        input_tensor = torch.tensor([[1.0]], dtype=torch.float32)
        tp_input = tp.Tensor(input_tensor, dtype=tp.float32)

        tp_output = tp.copy(tp_model(tp_input), tp.device("cpu"))

        torch_model.eval()
        with torch.no_grad():
            torch_output = torch_model(input_tensor)

        rtol_ = 2e-7
        assert torch.allclose(torch.from_dlpack(tp_output), torch_output, rtol=rtol_)
