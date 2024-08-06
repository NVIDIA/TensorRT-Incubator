#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import array
from dataclasses import dataclass
from typing import Optional, Sequence

from mlir_tensorrt.compiler.dialects import stablehlo
from mlir_tensorrt.compiler.dialects._ods_common import get_op_result_or_value
from mlir_tensorrt.compiler import ir

from tripy.flat_ir.ops.base import BaseFlatIROp
from tripy.backend.mlir.utils import is_any_dim_dynamic
import tripy.utils.utils as utils


def get_constant_value(arg) -> Optional[ir.DenseElementsAttr]:
    if isinstance(arg, ir.Value) and ir.OpResult.isinstance(arg):
        arg = ir.OpResult(arg).owner

    if isinstance(arg, ir.Operation):
        arg = arg.opview

    if isinstance(arg, stablehlo.ConstantOp):
        return arg.value

    return None


def _do_static_reshape(arg, new_shape: Sequence[int]):
    # If the input is a constant, then just reshape the constant.
    const_input = get_constant_value(arg)

    # For now, just handle i32 types since we don't have the convenience of numpy, we need
    # to handle each element type a different way using 'array.array'.
    if const_input and ir.IntegerType.get_signless(32) == const_input.type.element_type:
        new_type = ir.RankedTensorType.get(new_shape, const_input.type.element_type)
        new_attr = ir.DenseElementsAttr.get(array=array.array("i", const_input), type=new_type)
        return stablehlo.constant(new_attr)

    arg = get_op_result_or_value(arg)
    output_type = ir.RankedTensorType.get(new_shape, arg.type.element_type)
    return stablehlo.reshape(output_type, arg)


@dataclass(repr=False)
class ReshapeOp(BaseFlatIROp):
    def to_mlir(self, operands):
        return [_do_static_reshape(operands[0], self.outputs[0].to_mlir().shape)]


class DynamicReshapeOp(BaseFlatIROp):
    def to_mlir(self, operands):
        if is_any_dim_dynamic(operands[1]):
            # Tripy frontend does not have shape inference and stablehlo does not allow shape operand to be of dynamic shape.
            # Since DynamicReshapeOp was created internally by Tripy, we know the expected output rank. For dynamic_reshape operator, the shape of shape tensor is the same as output rank.
            new_shape = [self.outputs[0].rank]
            self.inputs[1].shape = new_shape
            operands[1].set_type(ir.RankedTensorType.get(new_shape, operands[1].type.element_type))

        # If the shape is a constant, then we can just do static reshape.
        const_shape_value = get_constant_value(operands[1])
        if const_shape_value:
            return [_do_static_reshape(operands[0], list(const_shape_value))]

        # build the dynamic reshape
        output = stablehlo.dynamic_reshape(
            result=self.outputs[0].to_mlir(), operand=operands[0], output_shape=operands[1]
        )
        return [output]
