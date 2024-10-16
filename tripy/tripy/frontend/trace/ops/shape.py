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
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy.utils import Result
from tripy import constraints
from tripy.common.datatype import DATA_TYPES


@dataclass(repr=False)
class Shape(BaseTraceOp):

    # always return a shape
    def infer_tensor_variants(self, inputs) -> Result:
        from tripy.frontend.shape import Shape as ShapeType

        return Result.ok([ShapeType])

    def infer_len(self):
        return [self.inputs[0].rank]

    def infer_rank(self):
        assert len(self.inputs) == 1, "ShapeOf operation should have exactly one input!"
        self.outputs[0].rank = 1

    def infer_dtypes(self):
        from tripy.common.datatype import int32

        self.outputs[0].dtype = int32

    def to_flat_ir(self, inputs, outputs):
        import tripy.frontend.trace.ops.utils as op_utils

        op_utils.get_shape_of_tensor(inputs[0], outputs[0])


@TENSOR_METHOD_REGISTRY("shape")
@property
@constraints.dtype_info(
    dtype_variables={
        "self_dtype": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
        "T2": ["int32"],
    },
    dtype_constraints={"self": "self_dtype", constraints.RETURN_VALUE: "T2"},
)
def shape(self: "tripy.Tensor") -> "tripy.Tensor":
    """
    Represents the shape of the tensor.

    Returns:
        A 1D tensor containing the shape of this tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.ones((8, 2))
        shape = input.shape

        assert np.array_equal(cp.from_dlpack(shape).get(), np.array([8, 2]))
    """
    return Shape.build([self])
