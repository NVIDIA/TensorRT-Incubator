#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from nvtripy.common.datatype import DATA_TYPES
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.frontend.ops._registry import register_tensor_method
from nvtripy.trace.ops.shape import Shape, GetDimensionSize
from nvtripy.types import ShapeLike
from nvtripy.utils import wrappers


@register_tensor_method("shape")
@property
@wrappers.interface(dtype_constraints={"self": "T1"}, dtype_variables={"T1": list(DATA_TYPES.keys())})
def shape(self: "nvtripy.Tensor") -> ShapeLike:
    """
    Represents the shape of the tensor.

    Returns:
        A sequence containing the shape of this tensor.

    .. code-block:: python
        :linenos:

        # doc: print-locals input shape
        input = tp.ones((8, 2))
        shape = input.shape

        assert shape == [8, 2]
    """

    # If the shape is statically known, we do not need to insert any operator calls.
    # However, if we are tracing, it might still be necessary to insert calls in the final program, so we will keep it.
    if all(dim >= 0 for dim in self.trace_tensor.shape) and not self.trace_tensor.is_compile_tracer:
        return self.trace_tensor.shape

    shape = op_utils.create_op(Shape, [self])
    return [
        op_utils.create_op(GetDimensionSize, [shape], dim=index, always_cast_to_dimension_size=True)
        for index in range(self.rank)
    ]
