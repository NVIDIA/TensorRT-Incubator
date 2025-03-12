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
from nvtripy import export
from nvtripy.frontend.ops.binary.create import create_binary_op
from nvtripy.trace.ops.binary import LogicalOr
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"lhs": "T1", "rhs": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["bool"]},
)
# TODO (pranavm): Add integration tests for this.
# TODO (pranavm): Use magic method (__or__?)
def logical_or(lhs: "nvtripy.Tensor", rhs: "nvtripy.Tensor") -> "nvtripy.Tensor":
    """
    Performs an elementwise logical OR.

    Args:
        lhs: The first input tensor.
        rhs: The second input tensor.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:

        a = tp.Tensor([True, False, False])
        b = tp.Tensor([False, True, False])
        output = tp.logical_or(a, b)

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([True, True, False]))
    """
    return create_binary_op(LogicalOr, lhs, rhs, cast_bool_to_int=False)
