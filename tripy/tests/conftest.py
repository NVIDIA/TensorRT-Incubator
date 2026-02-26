#
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
#

import os
import re
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


# TODO (pranavm): This is a temporary workaround and will be unnecessary with a future version of TensorRT.
def _drop_unused_entry_params(ptx: str, kernel_name: str) -> str:
    # Triton >= 3.6 may emit extra kernel entry parameters (e.g. scratch/runtime
    # plumbing) that are not referenced by the PTX body for simple kernels.
    #
    # TensorRT QDP currently launches this kernel correctly only when the PTX
    # entry signature matches the arguments we provide. To keep this test stable
    # across Triton versions, remove entry params that are never loaded from
    # (`ld.param ... [name]`) anywhere in the kernel body.
    lines = ptx.splitlines()

    entry_start = next((i for i, line in enumerate(lines) if f".entry {kernel_name}(" in line), None)
    if entry_start is None:
        return ptx

    entry_end = next((i for i in range(entry_start + 1, len(lines)) if lines[i].strip() == ")"), None)
    if entry_end is None:
        return ptx

    param_lines = lines[entry_start + 1 : entry_end]
    body = "\n".join(lines[entry_end + 1 :])

    def param_name(line: str):
        match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*,?\s*$", line)
        return match.group(1) if match and ".param" in line else None

    used = {name for line in param_lines if (name := param_name(line)) and re.search(rf"\[{re.escape(name)}\]", body)}

    filtered_params = [line for line in param_lines if (name := param_name(line)) is None or name in used]
    if len(filtered_params) == len(param_lines):
        return ptx

    for i in range(len(filtered_params) - 1, -1, -1):
        if ".param" in filtered_params[i]:
            filtered_params[i] = filtered_params[i].rstrip().rstrip(",")
            break

    updated_lines = lines[: entry_start + 1] + filtered_params + lines[entry_end:]
    return "\n".join(updated_lines)


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
        signature={"x_ptr": "*fp32", "n_elements": "i32", "y_ptr": "*fp32"},
        constexprs={"BLOCK_SIZE": block_size},
    )

    compiled_kernel = triton.compile(src)
    metadata = compiled_kernel.metadata

    N = inp0.shape_expr.numel()
    launch_params = trtp.KernelLaunchParams()

    launch_params.grid_x = trtp.cdiv(N, block_size)
    launch_params.block_x = metadata.num_warps * 32
    launch_params.shared_mem = metadata.shared

    extra_args = trtp.SymIntExprs(1)
    extra_args[0] = trtp.SymInt32(N)

    ptx = compiled_kernel.asm["ptx"]
    # Apply a compatibility normalization for Triton-generated PTX so this QDP
    # test works across Triton releases with different entry ABI shapes.
    ptx = _drop_unused_entry_params(ptx, metadata.name)

    return metadata.name, ptx, launch_params, extra_args
