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

from nvtripy.frontend.ops import utils as op_utils
from nvtripy.frontend.ops._registry import register_tensor_method
from nvtripy.trace.ops.matmul import MatrixMultiply
from nvtripy.utils import wrappers


# TODO (pranavm): Check if TRT supports int32 matmul or if we need to cast in the frontend.
@register_tensor_method("__matmul__")
@wrappers.interface(
    dtype_constraints={"self": "T1", "other": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16"]},
)
# TODO (pranavm): Add more examples for 1D/2D combinations and better testing, document output shape.
def __matmul__(self: "nvtripy.Tensor", other: "nvtripy.Tensor") -> "nvtripy.Tensor":
    """
    Performs matrix multiplication between two tensors.

    - If both tensors are 1D, a dot product is performed.
    - If both tensors are 2D, matrix multiplication is performed.
    - If either argument, but not both, is 1D, matrix-vector multiplication is performed.
    - If both tensors are 2D or higher dimensional and have differnt ranks, a dimension is inserted
        and batched matrix multiplication is performed with broadcast of relevant dimension.

    Args:
        self: Input tensor.
        other: The tensor by which to multiply.

    Returns:
        A new tensor.

    .. code-block:: python
        :linenos:

        a = tp.iota((2, 3), dtype=tp.float32)
        b = tp.iota((3, 2), dtype=tp.float32)

        output = a @ b
        assert np.array_equal(cp.from_dlpack(output).get(), cp.from_dlpack(a).get() @ cp.from_dlpack(b).get())
    """
    # Don't expand ranks if one of the inputs is rank 1:
    if self.rank != 1 and other.rank != 1:
        self, other = op_utils.match_ranks(self, other)
    return op_utils.create_op(MatrixMultiply, [self, other])
