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


import tensorrt as trt
import tensorrt.plugin as trtp

import cupy as cp

import nvtripy as tp

import triton
import triton.language as tl

from typing import Tuple, List, Union

@triton.jit
def add_kernel(x_ptr, n_elements, y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x + 1, mask=mask)

@trtp.register("example::elemwise_add_plugin")
def add_plugin_desc(inp0: trtp.TensorDesc, block_size: int) -> trtp.TensorDesc:
    return inp0.like()

@trtp.autotune("example::elemwise_add_plugin")
def add_plugin_autotune(inp0: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc]) -> List[trtp.AutoTuneCombination]:
    return [trtp.AutoTuneCombination("FP32|FP16, FP32|FP16")]

@trtp.aot_impl("example::elemwise_add_plugin")
def add_plugin_aot_impl(
    inp0: trtp.TensorDesc, block_size: int, outputs: Tuple[trtp.TensorDesc], tactic: int
) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:

    type_str = "fp32" if inp0.dtype == trt.float32 else "fp16"

    src = triton.compiler.ASTSource(
        fn=add_kernel,
        signature=f"*{type_str},i32,*{type_str}",
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


@trtp.register("example::circ_pad_plugin")
def circ_pad_plugin_desc(
    inp0: trtp.TensorDesc, pads: npt.NDArray[np.int32]
) -> trtp.TensorDesc:
    ndim = inp0.ndim
    out_desc = inp0.like()

    for i in range(np.size(pads) // 2):
        out_desc.shape_expr[ndim - i - 1] += int(pads[i * 2] + pads[i * 2 + 1])

    return out_desc

@trtp.autotune("example::circ_pad_plugin")
def circ_pad_plugin_autotune(
    inp0: trtp.TensorDesc,
    outputs: Tuple[trtp.TensorDesc],
) -> List[trtp.AutoTuneCombination]:

    return [trtp.AutoTuneCombination("FP32|FP16, FP32|FP16")]

@trtp.aot_impl("example::circ_pad_plugin")
def circ_pad_plugin_aot_impl(
    inp0: trtp.TensorDesc, pads: npt.NDArray[np.int32], outputs: Tuple[trtp.TensorDesc], tactic: int
) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:

    block_size = 256

    N = inp0.ndim
    all_pads = np.zeros((N * 2,), dtype=np.int32)
    inp_dims = inp0.shape_expr
    out_dims = outputs[0].shape_expr

    for i in range(np.size(pads) // 2):
        all_pads[N * 2 - 2 * i - 2] = pads[i * 2]
        all_pads[N * 2 - 2 * i - 1] = pads[i * 2 + 1]

    all_pads = all_pads.tolist()
    extra_args = trtp.SymIntExprs.from_tuple(
        [
            trtp.SymInt32(e)
            for e in [
                all_pads[0],
                all_pads[2],
                all_pads[4],
                all_pads[6],
                inp_dims[0],
                inp_dims[1],
                inp_dims[2],
                inp_dims[3],
                out_dims[1],
                out_dims[2],
                out_dims[3],
                inp_dims.numel(),
                out_dims.numel(),
            ]
        ]
    )

    type_str = "fp32" if inp0.dtype == trt.float32 else "fp16"

    src = triton.compiler.ASTSource(
        fn=circ_pad_kernel,
        signature=f"*{type_str},{','.join(['i32']*13)},*{type_str}",
        constants={
            "BLOCK_SIZE": block_size,
        },
    )

    compiled_kernel = triton.compile(src)
    launch_params = trtp.KernelLaunchParams()

    launch_params.grid_x = trtp.cdiv(out_dims.numel(), block_size)
    launch_params.block_x = compiled_kernel.metadata.num_warps * 32
    launch_params.shared_mem = compiled_kernel.metadata.shared

    return compiled_kernel.metadata.name.encode(), compiled_kernel.asm["ptx"].encode(), launch_params, extra_args

class TestQuickDeployablePlugin:
    def test_elemwise_add_plugin(self):
        inp = tp.iota((2, 2))
        out = tp.plugin(
            "example::elemwise_add_plugin",
            [inp],
            block_size = 256,
        )

        assert cp.allclose(cp.from_dlpack(out), cp.from_dlpack(inp + 1))

    def test_circ_pad_plugin(self):
        inp_shape = (10, 3, 32, 32)
        x = np.random.normal(size=inp_shape).astype(trt.nptype(trt.DataType.FLOAT))

        inp = tp.Tensor(x)
        pads = np.array((1, 1, 1, 1), dtype=np.int32)

        out = tp.plugin(
            "example::circ_pad_plugin",
            [inp],
            pads = pads,
        )

        ref = np.pad(x, [[0, 0], [0, 0], [pads[0], pads[1]], [pads[2], pads[3]]], "wrap")

        assert cp.allclose(cp.from_dlpack(out), ref)                                                                                                      

