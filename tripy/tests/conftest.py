#
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
#

import os
from typing import Tuple, Union

import nvtripy as tp
import pytest
import tensorrt as trt
import tensorrt.plugin as trtp
import torch
import triton
import triton.language as tl
from nvtripy.common.datatype import DATA_TYPES

HAS_FP8 = torch.cuda.get_device_capability() >= (8, 9)

skip_if_older_than_sm89 = pytest.mark.skipif(
    not HAS_FP8, reason="Some features (e.g. float8) are not available before SM89"
)

skip_if_older_than_sm80 = pytest.mark.skipif(
    torch.cuda.get_device_capability() < (8, 0), reason="Some features (e.g. bfloat16) are not available before SM80"
)

DATA_TYPE_TEST_CASES = [
    dtype if dtype not in [tp.float8] else pytest.param(tp.float8, marks=skip_if_older_than_sm89)
    for dtype in DATA_TYPES.values()
]


@pytest.fixture()
def tripy_virtualenv(virtualenv):
    """
    A virtual environment that inherits the PYTHONPATH from the host.
    """
    virtualenv.env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
    # The tensorrt_bindings package doesn't install correctly if there are cache entries.
    virtualenv.run("pip cache remove tensorrt*")
    return virtualenv


@triton.jit
def add_kernel(x_ptr, n_elements, y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x + 1, mask=mask)


@trtp.register("example::elemwise_add_plugin")
def add_plugin_desc(inp0: trtp.TensorDesc, block_size: int) -> trtp.TensorDesc:
    # QDPs should receive and return TRT data types
    assert inp0.dtype == trt.float32
    return inp0.like()


@trtp.aot_impl("example::elemwise_add_plugin")
def add_plugin_aot_impl(
    inp0: trtp.TensorDesc, block_size: int, outputs: Tuple[trtp.TensorDesc], tactic: int
) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:

    src = triton.compiler.ASTSource(
        fn=add_kernel,
        signature=f"*fp32,i32,*fp32",
        constants={
            "BLOCK_SIZE": block_size,
        },
    )

    compiled_kernel = triton.compile(src)

    N = inp0.shape_expr.numel()
    launch_params = trtp.KernelLaunchParams()

    launch_params.grid_x = trtp.cdiv(N, block_size)
    launch_params.block_x = compiled_kernel.metadata.num_warps * 32
    launch_params.shared_mem = compiled_kernel.metadata.shared

    extra_args = trtp.SymIntExprs(1)
    extra_args[0] = trtp.SymInt32(N)

    return compiled_kernel.metadata.name, compiled_kernel.asm["ptx"], launch_params, extra_args
