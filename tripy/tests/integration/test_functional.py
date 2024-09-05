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

import cupy as cp
import numpy as np
import pytest
import torch

import tripy as tp
from tripy.backend.mlir.compiler import Compiler
from tripy.backend.mlir.executor import Executor
from tripy.frontend.trace import Trace


class TestFunctional:
    @pytest.mark.parametrize("kind", ["cpu", "gpu"])
    def test_add_two_tensors(self, kind):
        module = cp if kind == "gpu" else np
        arr = module.array([2, 3], dtype=np.float32)
        a = tp.Tensor(arr, device=tp.device(kind))
        b = tp.Tensor(module.ones(2, dtype=module.float32), device=tp.device(kind))

        c = a + b
        out = c + c
        assert (cp.from_dlpack(out).get() == np.array([6.0, 8.0], dtype=np.float32)).all()

    @pytest.mark.parametrize(
        "dim_a, dim_b",
        [
            ((1, 3), (3, 3)),  # naive broadcast at 0th dim
            ((3, 3), (3, 1)),  # naive broadcast at 1sh dim of second operand
            ((1, 3, 1), (4, 3, 7)),  # broadcast at multiple dim of same operand
            ((1, 3, 7), (4, 3, 1)),  # broadcast at differnt dim of both operand
        ],
    )
    def test_static_broadcast_add_two_tensors(self, dim_a, dim_b):
        cp_a = cp.arange(np.prod(dim_a)).reshape(dim_a).astype(np.float32)
        cp_b = cp.arange(np.prod(dim_b)).reshape(dim_b).astype(np.float32)
        a = tp.Tensor(cp_a, device=tp.device("gpu"))
        b = tp.Tensor(cp_b, device=tp.device("gpu"))

        def func(a, b):
            c = a + b
            return c

        out = func(a, b)
        assert (cp.from_dlpack(out) == cp.array(cp_a + cp_b)).all()

    def test_multi_output_trace(self):
        arr = cp.ones(2, dtype=np.float32)
        a = tp.Tensor(arr)
        b = tp.Tensor(arr)
        c = a + b
        d = c + c
        trace = Trace([c, d])
        flat_ir = trace.to_flat_ir()

        compiler = Compiler()
        executor = Executor(compiler.compile(flat_ir.to_mlir()))
        out = executor.execute([out.device for out in flat_ir.outputs])
        assert (
            len(out) == 2
            and (cp.from_dlpack(out[0]) == cp.array([2.0, 2.0], dtype=np.float32)).all()
            and (cp.from_dlpack(out[1]) == cp.array([4.0, 4.0], dtype=np.float32)).all()
        )

    @pytest.mark.parametrize("kind", ["cpu", "gpu"])
    def test_weights_loading_from_torch(self, kind):
        with torch.no_grad():
            if kind == "gpu":
                inp = torch.arange(4, dtype=torch.float32, device=torch.device("cuda")).reshape(*(2, 2))
            else:
                inp = torch.arange(4, dtype=torch.float32).reshape(*(2, 2))

            if kind == "gpu":
                torch_linear = torch.nn.Linear(2, 3).to("cuda")
            else:
                torch_linear = torch.nn.Linear(2, 3)
            torch_out = torch_linear(inp)

            tripy_linear = tp.Linear(2, 3)
            if kind == "gpu":
                tripy_linear.weight = tp.Parameter(
                    tp.Tensor(torch_linear.weight.detach().to("cuda"), device=tp.device(kind))
                )
                tripy_linear.bias = tp.Parameter(
                    tp.Tensor(torch_linear.bias.detach().to("cuda"), device=tp.device(kind))
                )
            else:
                tripy_linear.weight = tp.Parameter(tp.Tensor(torch_linear.weight.detach(), device=tp.device(kind)))
                tripy_linear.bias = tp.Parameter(tp.Tensor(torch_linear.bias.detach(), device=tp.device(kind)))

            tripy_out = tripy_linear(tp.Tensor(inp, device=tp.device(kind)))
            assert tp.allclose(tripy_out, tp.Tensor(torch_out))


class TestCopyFunctional:
    @pytest.mark.parametrize("src", ["cpu", "gpu"])
    @pytest.mark.parametrize("dst", ["cpu", "gpu"])
    def test_single_copy(self, src, dst):
        a = tp.Tensor([1, 2], device=tp.device(src))
        out = tp.copy(a, tp.device(dst))
        assert out.tolist() == [1, 2]
        assert out.device.kind == dst

    def test_multiple_copy_1(self):
        a = tp.Tensor([1, 2])
        a = tp.copy(a, tp.device("gpu"))
        out = tp.copy(a, tp.device("cpu"))
        assert out.tolist() == [1, 2]
        assert out.device.kind == "cpu"

    def test_multiple_copy_2(self):
        a = tp.Tensor([1, 2])
        a = tp.copy(a, tp.device("cpu"))
        out = tp.copy(a, tp.device("gpu"))
        assert out.tolist() == [1, 2]
        assert out.device.kind == "gpu"


class TestConversionToTripyType:
    @pytest.mark.parametrize(
        "reverse_direction",
        [False, True],
    )
    @pytest.mark.parametrize(
        "input0",
        [cp.ones((2, 3), dtype=cp.float32), cp.ones((3,), dtype=np.float32)],
    )
    @pytest.mark.parametrize(
        "input1",
        [
            [
                4.0,
            ],
            (5.0,),
            cp.array([4.0], dtype=cp.float32),
            cp.ones((1, 3), dtype=cp.float32),
            torch.Tensor([[4.0]]),
        ],
    )
    def test_element_wise_prod(self, reverse_direction, input0, input1):
        a = tp.Tensor(input0)
        if isinstance(input1, torch.Tensor):
            input1 = input1.to("cuda")
        if reverse_direction:
            out = tp.Tensor(input1) * a
            input0, input1 = input1, input0
        else:
            out = a * tp.Tensor(input1)
        assert cp.array_equal(cp.from_dlpack(out), cp.array(input0) * cp.array(input1))
