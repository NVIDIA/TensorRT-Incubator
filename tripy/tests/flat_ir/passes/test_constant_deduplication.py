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
from tripy.flat_ir.ops import ConstantOp
from tripy.flat_ir.tensor import FlatIRTensor
from tripy.common.device import device
from tripy.common.datatype import int32
from tripy.flat_ir.passes.constant_deduplicate import ConstantDeduplicationPass


class MockOp:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs


def create_mock_flat_ir():
    flat_ir = FlatIR()

    # Create constant tensors
    const1 = FlatIRTensor.build(shape=[2], rank=1, dtype=int32, reason_details="", device=device("gpu"))
    const2 = FlatIRTensor.build(shape=[2], rank=1, dtype=int32, reason_details="", device=device("gpu"))
    const3 = FlatIRTensor.build(shape=[3], rank=1, dtype=int32, reason_details="", device=device("gpu"))

    # Create constant ops
    op1 = ConstantOp.build([], [const1], data=[1, 2])
    op2 = ConstantOp.build([], [const2], data=[1, 2])  # Duplicate of op1
    op3 = ConstantOp.build([], [const3], data=[3, 4, 5])

    # Create a mock op that uses the constants
    result_tensor = FlatIRTensor.build(shape=[2], rank=1, dtype=int32, reason_details="", device=device("gpu"))
    mock_op = MockOp([const1, const2, const3], [result_tensor])

    # Add ops to FlatIR
    flat_ir.ops = [op1, op2, op3, mock_op]
    flat_ir.outputs = [result_tensor]

    return flat_ir


def test_constant_deduplication_pass():
    flat_ir = create_mock_flat_ir()
    original_op_count = len(flat_ir.ops)

    # Apply the constant deduplication pass
    pass_instance = ConstantDeduplicationPass()
    pass_instance.run(flat_ir)

    # Verify that one constant op was removed
    assert len(flat_ir.ops) == original_op_count - 1, "One constant op should have been removed"

    # Verify that only two ConstantOps remain
    constant_ops = [op for op in flat_ir.ops if isinstance(op, ConstantOp)]
    assert len(constant_ops) == 2, "There should be only two ConstantOps after deduplication"

    # Verify that the remaining ConstantOps have different data
    constant_data = [tuple(op.data) for op in constant_ops]
    assert len(set(constant_data)) == 2, "The remaining ConstantOps should have different data"

    # Verify that the mock op now uses the same tensor for its first two inputs
    mock_op = [op for op in flat_ir.ops if isinstance(op, MockOp)][0]
    assert mock_op.inputs[0] == mock_op.inputs[1], "The mock op should use the same tensor for its first two inputs"
    assert mock_op.inputs[0] != mock_op.inputs[2], "The mock op should still have a different third input"

    # Verify that the output tensor is still in the graph
    assert flat_ir.outputs[0] in [op.outputs[0] for op in flat_ir.ops], "The output tensor should still be in the graph"
