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
import pytest
import re
from dataclasses import dataclass
from typing import List, Optional

from tripy.flat_ir.flat_ir import FlatIR
from tripy.flat_ir.ops.base import FlatIRFunction, BaseFlatIROp
from tripy.flat_ir.ops import ConstantOp
from tripy.flat_ir.tensor import FlatIRTensor
from tripy.common.device import device
from tripy.common.datatype import float32, int32


@dataclass(repr=False, eq=False)
class MockOp(BaseFlatIROp):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.trace_input_names = []
        self.trace_output_names = []
        for output in outputs:
            output.producer = self

    def __eq__(self, other):
        return True

    def to_mlir(self, operands):
        assert "Not implemented"


def test_is_structurally_equivalent():
    """Test the structural equivalence of two FlatIR functions."""
    flat_ir = FlatIR()

    def create_tensor(reason_details: str, name: Optional[str] = None) -> FlatIRTensor:
        """Create and register a FlatIRTensor."""
        t = FlatIRTensor.build(
            shape=[3],
            rank=1,
            dtype=float32,
            device=device("gpu"),
            reason_details=reason_details,
        )
        if name:
            t.name = name
        flat_ir.register_tensor(t)
        return t

    def create_function(
        name: str,
        input_tensor: FlatIRTensor,
        output_tensors: List[FlatIRTensor],
    ) -> FlatIRFunction:
        """Create a FlatIRFunction with associated operations."""
        callee_input = input_tensor.clone(reason_details=f"{name} input cloned from {input_tensor}")
        callee_outputs = [out.clone(reason_details=f"{name} output cloned from {out}") for out in output_tensors]

        flat_ir.register_tensor(callee_input)
        setattr(callee_input, "caller_tensor", input_tensor)

        for callee_out, original_out in zip(callee_outputs, output_tensors):
            flat_ir.register_tensor(callee_out)
            setattr(callee_out, "caller_tensor", original_out)

        func = FlatIRFunction(name, [callee_input], callee_outputs)
        mock_op = MockOp([callee_input], [callee_outputs[0]])
        const_op = ConstantOp.build([], [callee_outputs[1]], data=[3, 4, 5])
        callee_outputs[1].producer = const_op

        func.ops.extend([mock_op, const_op])
        for out in output_tensors:
            out.producer = func

        return func

    # Create main tensors
    input_tensor = create_tensor("Function 1 input", "main_input_tensor")
    intermediates = [create_tensor(f"Function 1 output {i}", f"intermediate_tensor_{i}") for i in range(2)]
    outputs = [create_tensor(f"Function 2 output {i}", f"main_output_tensor_{i}") for i in range(2)]

    # Create two structurally equivalent functions
    func_1 = create_function("Func1", input_tensor, intermediates)
    func_2 = create_function("Func2", intermediates[0], outputs)

    # Assert structural equivalence
    assert func_1.is_structurally_equivalent(func_2)

    # Set up FlatIR inputs and outputs
    flat_ir.inputs = [input_tensor]
    flat_ir.outputs = outputs

    # Integrate subgraphs
    for in_tensor, out_tensors in [(input_tensor, intermediates), (intermediates[0], outputs)]:
        flat_ir.integrate_subgraph([in_tensor], out_tensors)

    flat_ir_str = str(flat_ir)

    # Check Func1 structure
    func_pattern = re.compile(r"function\s+Func1\s*\(\s*\w+:.*?\)\s*->\s*\(.*?\)\s*{.*?return.*?}", re.DOTALL)
    assert func_pattern.search(flat_ir_str), "Function Func1 structure is incorrect"

    # Check Main Function structure
    main_pattern = re.compile(
        r"Main Function:.*?inputs:.*?=\s*function Func1.*?=\s*function Func1.*?outputs:", re.DOTALL
    )
    assert main_pattern.search(flat_ir_str), "Main Function structure is incorrect"

    print("All assertions passed. Function structures are correct.")
