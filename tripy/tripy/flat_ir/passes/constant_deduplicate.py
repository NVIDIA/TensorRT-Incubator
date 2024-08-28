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

from typing import List, Dict, Tuple, Any, Union
from tripy.flat_ir.ops import ConstantOp
from tripy.flat_ir.tensor import FlatIRTensor
from tripy.flat_ir.flat_ir import FlatIRPass, FlatIR


class ConstantDeduplicationPass(FlatIRPass):
    """
    Optimizes the flatIR such that previously defined constant ops are reused.
    """

    def run(self, flat_ir: FlatIR):
        constant_map: Dict[Any, FlatIRTensor] = {}
        new_ops = []
        for op in flat_ir.ops:
            if isinstance(op, ConstantOp):
                constant_key = self._get_constant_key(op)
                if constant_key in constant_map:
                    # Reuse existing constant
                    existing_tensor = constant_map[constant_key]
                    self._replace_tensor_uses(op, existing_tensor, flat_ir)
                else:
                    # New unique constant, add to map
                    constant_map[constant_key] = op.outputs[0]
                    new_ops.append(op)
            else:
                new_ops.append(op)

        # Update FlatIR with optimized ops
        flat_ir.ops = new_ops

    def _get_constant_key(self, op):
        from mlir_tensorrt.runtime._mlir_libs._api import MemRefValue
        from tripy.utils.utils import list_to_tuple

        if isinstance(op.data, MemRefValue):
            from tripy.backend.mlir.memref import tolist

            l = tolist(op.data)
            data = tuple(list_to_tuple(l if isinstance(l, List) else [l]))
        elif isinstance(op.data, int) or isinstance(op.data, float) or isinstance(op.data, bool):
            data = list_to_tuple(
                op.data,
            )
        else:
            data = list_to_tuple(op.data)

        # Create a unique key for the constant based on its data and type
        return (data, op.outputs[0].dtype, list_to_tuple(op.outputs[0].shape))

    def _replace_tensor_uses(self, op_to_remove, exisiting_tensor: FlatIRTensor, flat_ir: FlatIR):
        old_tensor = op_to_remove.outputs[0]
        # Replace all uses of the old tensor with the new tensor
        for op in flat_ir.ops:
            if op != op_to_remove:
                op.inputs = [exisiting_tensor if input == old_tensor else input for input in op.inputs]

        # Update outputs if necessary
        flat_ir.outputs = [exisiting_tensor if output == old_tensor else output for output in flat_ir.outputs]

        # Remove the old tensor from the tensor map
        if old_tensor.name in flat_ir.tensor_map:
            del flat_ir.tensor_map[old_tensor.name]


# # Attach the pass to FlatIR
FlatIR.attach_pass(ConstantDeduplicationPass)
