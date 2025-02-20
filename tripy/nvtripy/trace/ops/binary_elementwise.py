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

from dataclasses import dataclass

import nvtripy.trace.ops.utils as op_utils
from nvtripy.common import datatype
from nvtripy.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class BinaryElementwise(BaseTraceOp):
    class Kind:
        SUM = " + "
        SUB = " - "
        POW = " ** "
        MUL = " * "
        DIV = " / "
        FLOOR_DIV = " // "
        MOD = " % "
        MAXIMUM = "maximum"
        MINIMUM = "minimum"

    kind: str

    def __str__(self):
        if self.kind.startswith(" "):
            op_str = self.kind.join([inp.name for inp in self.inputs])
        else:
            op_str = f"{self.kind}({self.inputs[0].name}, {self.inputs[1].name})"
        return f"{self.outputs[0].name} = {op_str}"

    infer_rank = op_utils.InferRankPolicies.max_of_inputs()

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def broadcast_inputs(self, inputs):
        from nvtripy.flat_ir.tensor import FlatIRTensor

        rank = max(inputs[0].rank, inputs[1].rank)
        with FlatIRTensor.context([f"expand the inputs of '{self.kind.strip()}' to have the same rank"]):
            broadcasted_input_0 = op_utils.expand_rank_of_tensor(inputs[0], rank - inputs[0].rank)
            broadcasted_input_1 = op_utils.expand_rank_of_tensor(inputs[1], rank - inputs[1].rank)

        with FlatIRTensor.context([f"broadcast the inputs of '{self.kind.strip()}' to compatible shapes"]):
            shape_of_input0 = op_utils.get_shape_of_tensor(broadcasted_input_0)
            shape_of_input1 = op_utils.get_shape_of_tensor(broadcasted_input_1)

            # Compute element-wise max of input shapes to get the desired output shape.
            output_shape_tensor = op_utils.compute_shape_of_broadcast(
                shape_of_input0,
                shape_of_input1,
                rank,
                shape1_name=f"the shape of the first input {shape_of_input0}",
                shape2_name=f"the shape of the second input {shape_of_input1}",
            )

            with FlatIRTensor.context([f"broadcasting the inputs of '{self.kind.strip()}'"]):
                broadcasted_input_0 = op_utils.insert_broadcast(
                    broadcasted_input_0,
                    out_rank=rank,
                    shape_of_target_tensor=output_shape_tensor,
                    tensor_details=f"left operand",
                )

                broadcasted_input_1 = op_utils.insert_broadcast(
                    broadcasted_input_1,
                    out_rank=rank,
                    shape_of_target_tensor=output_shape_tensor,
                    tensor_details=f"right operand",
                )

        return [broadcasted_input_0, broadcasted_input_1]

    def to_flat_ir(self, inputs, outputs):
        from nvtripy.flat_ir.ops import AddOp, DivideOp, FloorOp, MaxOp, MinOp, MulOp, PowOp, SubtractOp
        from nvtripy.flat_ir.tensor import FlatIRTensor

        broadcasted_inputs = self.broadcast_inputs(inputs)

        if self.kind == BinaryElementwise.Kind.FLOOR_DIV:
            # First apply DivideOp
            divide_out = FlatIRTensor.build(
                shape=outputs[0].shape,
                rank=outputs[0].rank,
                dtype=outputs[0].dtype,
                device=outputs[0].device,
                reason_details=["Intermediate output of division operator for FLOOR_DIV operation."],
            )
            DivideOp.build(broadcasted_inputs, [divide_out])
            # Then apply FloorOp to the result of the division
            FloorOp.build([divide_out], outputs)
        elif self.kind == BinaryElementwise.Kind.MOD:
            # Step 1: Perform DivideOp
            divide_out = FlatIRTensor.build(
                shape=outputs[0].shape,
                rank=outputs[0].rank,
                dtype=outputs[0].dtype,
                device=outputs[0].device,
                reason_details=["Intermediate output of division operator for MOD operation."],
            )
            DivideOp.build(broadcasted_inputs, [divide_out])

            # Step 2: Apply FloorOp
            floor_out = FlatIRTensor.build(
                shape=outputs[0].shape,
                rank=outputs[0].rank,
                dtype=outputs[0].dtype,
                device=outputs[0].device,
                reason_details=["Intermediate output of Floor operation for MOD operation."],
            )
            FloorOp.build([divide_out], [floor_out])

            # Step 3: Multiply divisor with floored division result (FloorOp output)
            multiply_out = FlatIRTensor.build(
                shape=outputs[0].shape,
                rank=outputs[0].rank,
                dtype=outputs[0].dtype,
                device=outputs[0].device,
                reason_details=["Intermediate output of Multiply operation for MOD operation."],
            )
            MulOp.build([broadcasted_inputs[1], floor_out], [multiply_out])

            # Step 4: Subtract result from dividend (broadcasted_inputs[0]) to get modulus
            SubtractOp.build([broadcasted_inputs[0], multiply_out], outputs)
        else:
            OpType = {
                BinaryElementwise.Kind.SUM: AddOp,
                BinaryElementwise.Kind.POW: PowOp,
                BinaryElementwise.Kind.MUL: MulOp,
                BinaryElementwise.Kind.SUB: SubtractOp,
                BinaryElementwise.Kind.DIV: DivideOp,
                BinaryElementwise.Kind.MAXIMUM: MaxOp,
                BinaryElementwise.Kind.MINIMUM: MinOp,
                BinaryElementwise.Kind.FLOOR_DIV: DivideOp,
            }[self.kind]
            OpType.build(broadcasted_inputs, outputs)


@dataclass(repr=False)
class Comparison(BinaryElementwise):
    class Kind:
        class KindElem(str):
            def __new__(cls, content, compare_direction):
                instance = super().__new__(cls, content)
                instance.compare_direction = compare_direction
                return instance

        LESS = KindElem(" < ", "LT")
        LESS_EQUAL = KindElem(" <= ", "LE")
        EQUAL = KindElem(" == ", "EQ")
        NOT_EQUAL = KindElem(" != ", "NE")
        GREATER_EQUAL = KindElem(" >= ", "GE")
        GREATER = KindElem(" > ", "GT")

    kind: Kind.KindElem

    def infer_dtypes(self):
        self.outputs[0].dtype = datatype.bool

    def to_flat_ir(self, inputs, outputs):
        from nvtripy.flat_ir.ops import CompareOp

        inputs = self.broadcast_inputs(inputs)
        CompareOp.build(inputs, outputs, compare_direction=self.kind.compare_direction)
