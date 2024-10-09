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
import numpy as np
import torch
from tests.helper import TRIPY_TO_NUMPY
from tests.performance.conftest import PerfParam, perf_fixture

import tripy as tp


@perf_fixture(
    params=[
        PerfParam(tp.float32, 1.25),
        PerfParam(tp.float16),
    ]
)
def linear_block(tripy_dtype, torch_dtype):

    NUM_LAYERS = 15

    class LinearBlock(tp.Module):
        def __init__(self):
            self.layers = [tp.Linear(256, 256, bias=False, dtype=tripy_dtype) for _ in range(NUM_LAYERS)]
            for layer in self.layers:
                # Adjust the weights to prevent FP16 overflows:
                weight = np.tile(np.array([[-1, 1], [1, -1]], dtype=TRIPY_TO_NUMPY[tripy_dtype]), (128, 128))
                layer.weight = tp.Parameter(weight)

        def __call__(self, input):
            for layer in self.layers:
                input = layer(input)
            return input

    class TorchLinearBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(256, 256, bias=False, dtype=torch_dtype, device=torch.device("cuda"))
                    for _ in range(NUM_LAYERS)
                ]
            )

        def forward(self, input):
            for layer in self.layers:
                input = layer(input)
            return input

    tripy_block = LinearBlock()
    torch_block = TorchLinearBlock()
    input_infos = {"input": tp.InputInfo(shape=(1024, 256), dtype=tripy_dtype)}
    return tripy_block, torch_block, input_infos
