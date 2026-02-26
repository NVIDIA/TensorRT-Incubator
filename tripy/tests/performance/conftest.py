# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import re
import sys
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

    def _sanitize_name(name: str) -> str:
        return re.sub(r"[^0-9a-zA-Z_]", "_", name)

    def perf_fixture_impl(func):
        module = sys.modules[func.__module__]

        for index, selected_param in enumerate(params):
            dtype = selected_param.dtype
            dtype_name = _sanitize_name(getattr(dtype, "name", str(dtype)))
            fixture_name = f"{func.__name__}_{dtype_name}_{index}"
            PERF_CASES.append(fixture_name)

            @pytest.fixture(scope="session", name=fixture_name)
            def wrapped(selected_param=selected_param):
                dtype, perf_threshold = selected_param.dtype, selected_param.perf_threshold
                tripy_module, torch_module, input_infos = func(dtype, helper.TORCH_DTYPES[dtype])

                torch_state_dict = {key: torch.from_dlpack(value) for key, value in tripy_module.state_dict().items()}
                torch_module.load_state_dict(torch_state_dict)

                tripy_compiled = tp.compile(tripy_module, kwargs=input_infos)

                inputs = {
                    key: tp.iota(input_info.shape_bounds.opt, dtype=dtype) for key, input_info in input_infos.items()
                }
                for tensor in inputs.values():
                    tensor.eval()

                torch_compiled = torch.compile(torch_module)

                return tripy_compiled, torch_compiled, inputs, perf_threshold

            module.__dict__[fixture_name] = wrapped

        return func

    return perf_fixture_impl
