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
from dataclasses import dataclass
from typing import List

import pytest
import torch
from tests import helper

import nvtripy as tp

PERF_CASES = []


@dataclass
class PerfParam:
    dtype: tp.dtype
    """Data type to use"""
    perf_threshold: float = 1.05
    """
    A multiplier indicating how much faster Tripy should be compared to Torch.
    For example, 1.05 would mean that Tripy should be 5% faster than Torch.
    """


def perf_fixture(params: List[PerfParam]):

    def perf_fixture_impl(func):
        PERF_CASES.append(pytest.lazy_fixture(func.__qualname__))

        @pytest.fixture(params=params, scope="session", ids=lambda param: param.dtype)
        def wrapped(request):
            dtype, perf_threshold = request.param.dtype, request.param.perf_threshold
            tripy_module, torch_module, input_infos = func(dtype, helper.TORCH_DTYPES[dtype])

            torch_state_dict = {key: torch.from_dlpack(value) for key, value in tripy_module.state_dict().items()}
            torch_module.load_state_dict(torch_state_dict)

            tripy_compiled = tp.compile(tripy_module, kwargs=input_infos)

            inputs = {key: tp.iota(input_info.shape_bounds.opt, dtype=dtype) for key, input_info in input_infos.items()}
            for tensor in inputs.values():
                tensor.eval()

            torch_compiled = torch.compile(torch_module)

            return tripy_compiled, torch_compiled, inputs, perf_threshold

        return wrapped

    return perf_fixture_impl
