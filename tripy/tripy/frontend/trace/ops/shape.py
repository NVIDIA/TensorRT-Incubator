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

from tripy import constraints
from tripy.common.datatype import DATA_TYPES
from tripy.frontend.ops.registry import register_tensor_method
from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy.types import ShapeLike


@dataclass(repr=False)
class GetDimensionSize(BaseTraceOp):
    dim: int

    def infer_rank(self):
        self.outputs[0].rank = 0

    def infer_dtypes(self):
        from tripy.common.datatype import int32

        self.outputs[0].dtype = int32

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import GetDimensionSizeOp

        GetDimensionSizeOp.build(inputs, outputs, self.dim)


@register_tensor_method("shape")
@property
@constraints.interface(dtype_constraints={"self": "T1"}, variables={"T1": list(DATA_TYPES.keys())})
def shape(self: "tripy.Tensor") -> ShapeLike:
    """
    Represents the shape of the tensor.

    Returns:
        A shape tensor containing the shape of this tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        # doc: print-locals input shape
        input = tp.ones((8, 2))
        shape = input.shape

        assert shape == [8, 2]
    """

    # If the shape is statically known, we do not need to insert any operator calls.
    # However, if we are tracing, it might still be necessary to insert calls in the final program, so we will keep it.
    if all(dim >= 0 for dim in self.trace_tensor.shape) and not self.trace_tensor.is_compile_tracer:
        return self.trace_tensor.shape

    return [GetDimensionSize.build([self], dim=index, always_cast_to_dimension_size=True) for index in range(self.rank)]
