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

from dataclasses import dataclass

from tripy import export, utils
from tripy.common.datatype import int32
from tripy.frontend.trace.ops.base import BaseTraceOp
import tripy.frontend.trace.ops.utils as op_utils


@dataclass(repr=False)
class Unsqueeze(BaseTraceOp):

    dim: int

    # the result will not be rank 1 and so can't be a shape but we may want to unsqueeze shapes
    infer_shape_output_idxs = op_utils.ShapeOutputIdxPolicies.never_return_shape

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def infer_rank(self):
        self.outputs[0].rank = self.inputs[0].rank + 1

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import DynamicBroadcastOp

        broadcast_dim = list(range(inputs[0].rank))
        for idx in range(len(broadcast_dim)):
            if idx >= self.dim:
                broadcast_dim[idx] += 1

        DynamicBroadcastOp.build(
            [inputs[0], inputs[1]],
            [outputs[0]],
            broadcast_dim=broadcast_dim,
        )


# Two operand unsqueeze op to ensure that Trace op is 1:1 with Python code (for error messaging).
def unsqueeze_two_operand(input, result_shape, dim):
    return Unsqueeze.build([input, result_shape], dim)


@export.public_api(document_under="operations/functions")
def unsqueeze(input: "tripy.Tensor", dim: int) -> "tripy.Tensor":
    """
    Returns a new tensor with the contents of the input tensor with a
    singleton dimension inserted at the specified position.

    Args:
        input: The input tensor.
        dim: index to insert the singleton dimension.

    Returns:
        A new tensor of the same data type as the input tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.iota((2, 2), dtype=tp.float32)
        output = tp.unsqueeze(input, 1)

        assert np.array_equal(cp.from_dlpack(output).get(), np.expand_dims(cp.from_dlpack(input).get(), 1))
    """
    from tripy.frontend.trace.ops.concatenate import concatenate

    from tripy.frontend import Shape, Tensor

    if dim < 0:
        dim = dim + input.rank + 1

    # Add specical case for rank 0 since tensor.shape is not supported when rank is 0.
    if input.rank == 0:
        result_shape = Shape([1])
    else:
        input_shape = input.shape
        result_shape = concatenate([input_shape[:dim], Shape([1]), input_shape[dim:]], dim=0)
    return unsqueeze_two_operand(input, result_shape, dim)
