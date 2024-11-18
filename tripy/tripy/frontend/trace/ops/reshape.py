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

import math
from dataclasses import dataclass

from tripy import constraints, export
from tripy.common.exception import raise_error
from tripy.frontend import utils as frontend_utils
from tripy.frontend.trace.ops import utils as op_utils
from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy.types import ShapeLike


@dataclass(repr=False)
class Reshape(BaseTraceOp):

    output_rank: int

    infer_rank = op_utils.InferRankPolicies.same_as_shape_of_shape_input(1)

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import DynamicReshapeOp

        DynamicReshapeOp.build(inputs, outputs)


def infer_dimensions(input: "tripy.Tensor", shape: ShapeLike) -> ShapeLike:

    num_unknown_dims = len([dim for dim in shape if op_utils.is_minus_one(dim)])
    if num_unknown_dims > 1:
        raise_error(f"The new shape can have at most one inferred dimension (denoted by -1)", [f"Got shape: {shape}."])

    if num_unknown_dims == 1:
        input_shape = input.shape
        input_volume = math.prod(input_shape)
        known_dims_volume = math.prod(dim for dim in shape if not op_utils.is_minus_one(dim))
        inferred_dim = input_volume / known_dims_volume

        shape = [inferred_dim if op_utils.is_minus_one(dim) else dim for dim in shape]

    return {"shape": shape}


@export.public_api(document_under="operations/functions")
@frontend_utils.convert_to_tensors(preprocess_args=infer_dimensions)
@constraints.dtypes(
    constraints={"input": "T1", constraints.RETURN_VALUE: "T1"},
    variables={"T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"]},
)
def reshape(input: "tripy.Tensor", shape: ShapeLike) -> "tripy.Tensor":
    """
    Returns a new tensor with the contents of the input tensor in the specified shape.

    Args:
        input: The input tensor.
        shape: The desired compatible shape. If a shape dimension is -1, its value
            is inferred based on the other dimensions and the number of elements in the input.
            Atmost one dimension can be -1.

    Returns:
        A new tensor with the specified shape.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.iota((2, 3), dtype=tp.float32)
        output = tp.reshape(input, (1, 6))

        assert np.array_equal(cp.from_dlpack(output).get(), np.reshape(cp.from_dlpack(input).get(), (1, 6)))
    """
    return Reshape.build([input, shape], None)
