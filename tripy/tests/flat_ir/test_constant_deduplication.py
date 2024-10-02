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
from tripy.flat_ir.flat_ir import FlatIR
from tripy.flat_ir.function import FlatIRFunction
from tripy.flat_ir.ops import ConstantOp
from tripy.flat_ir.tensor import FlatIRTensor
from tripy.common.device import device
from tripy.common.datatype import int32


class MockOp:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        for output in outputs:
            output.producer = self


def create_subgraph(config):
    # Create constant tensors and ops
    const1 = FlatIRTensor.build(shape=[2], rank=1, dtype=int32, reason_details="", device=device("gpu"))
    op1 = ConstantOp.build([], [const1], data=[1, 2])
    const1.producer = op1

    # Duplicate of const1
    const2 = FlatIRTensor.build(shape=[2], rank=1, dtype=int32, reason_details="", device=device("gpu"))
    op2 = ConstantOp.build([], [const2], data=[1, 2])
    const2.producer = op2

    const3 = FlatIRTensor.build(shape=[3], rank=1, dtype=int32, reason_details="", device=device("gpu"))
    op3 = ConstantOp.build([], [const3], data=[3, 4, 5])
    const3.producer = op3

    # Create a mock op that uses the constants
    result_tensor = FlatIRTensor.build(shape=[2], rank=1, dtype=int32, reason_details="", device=device("gpu"))
    mock_op = MockOp([const1, const2, const3], [result_tensor])

    if config == "func":
        # Create a function with no inputs and a single output
        func_result_tensor = FlatIRTensor.build(shape=[2], rank=1, dtype=int32, reason_details="", device=device("gpu"))
        setattr(result_tensor, "caller_tensor", func_result_tensor)
        func = FlatIRFunction("MockFunc", [], [result_tensor])
        func_result_tensor.producer = func

        # Insert all operations in a function
        func.ops = [op1, op2, op3, mock_op]

        # Return function result tensor i.e. output of a function call
        return [], [func_result_tensor]

    return [], [result_tensor]


@pytest.mark.parametrize("config", ["main", "func"])
def test_integrate_subgraph_constant_deduplication(config):
    flat_ir = FlatIR()
    inputs, outputs = create_subgraph(config)

    # Integrate the subgraph
    flat_ir.integrate_subgraph(inputs, outputs)

    # Verify that only two ConstantOps remain
    ops = flat_ir.ops[0].ops if isinstance(flat_ir.ops[0], FlatIRFunction) else flat_ir.ops

    constant_ops = [op for op in ops if isinstance(op, ConstantOp)]
    assert len(constant_ops) == 2, "There should be only two ConstantOps after integration"

    # Verify that the remaining ConstantOps have different data
    constant_data = [tuple(op.data) for op in constant_ops]
    assert constant_data
    assert len(set(constant_data)) == 2, "The remaining ConstantOps should have different data"

    # Check for the specific expected data
    expected_data = {(1, 2), (3, 4, 5)}
    assert set(constant_data) == expected_data, f"Expected constant data {expected_data}, but got {set(constant_data)}"

    # Verify that the mock op now uses the same tensor for its first two inputs
    mock_op = [op for op in ops if isinstance(op, MockOp)][0]
    assert mock_op.inputs[0] is mock_op.inputs[1], "The mock op should use the same tensor for its first two inputs"
    assert mock_op.inputs[0] is not mock_op.inputs[2], "The mock op should still have a different third input"

    if config == "main":
        # Verify that tensor replacements were applied
        assert len(flat_ir.tensor_replacements) > 0, "There should be tensor replacements after integration"

        # Verify that the constant map has the correct number of entries
        assert len(flat_ir.constant_map) == 2, "Constant map should have 2 entries"
