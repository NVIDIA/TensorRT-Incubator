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

from nvtripy.common.exception import raise_error
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.frontend.ops._registry import register_tensor_method
from nvtripy.trace.ops.matmul import MatrixMultiply
from nvtripy.utils import wrappers


@register_tensor_method("__matmul__")
@wrappers.interface(
    dtype_constraints={"self": "T1", "other": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16"]},
)
def __matmul__(self: "nvtripy.Tensor", other: "nvtripy.Tensor") -> "nvtripy.Tensor":
    """
    Performs matrix multiplication between two tensors.

    - If both tensors are 1D, a dot product is performed. The output is a scalar.

    - If either argument, but not both, is 1D, matrix-vector multiplication is performed:
        - For inputs of shape :math:`(M, N)` and :math:`(N,)`, the output will have shape :math:`(M,)`.
        - For inputs of shape :math:`(N,)` and :math:`(N, K)`, the output will have shape :math:`(K,)`.

    - If both tensors are 2D, matrix-matrix multiplication is performed.
        For inputs of shape :math:`(M, N)` and :math:`(N, K)`, the output will have shape :math:`(M, K)`.

    - If the tensor has more than 2 dimensions, it is treated as a stack of matrices.
        If the ranks differ for tensors with 2 or more dimensions, dimensions are prepended until the ranks match.
        The first :math:`N-2` dimensions will be broacasted if required.

    Args:
        self: Input tensor.
        other: The tensor by which to multiply.

    Returns:
        A new tensor.

    .. code-block:: python
        :linenos:
        :caption: Dot Product

        a = tp.iota((3,), dtype=tp.float32)
        b = tp.iota((3,), dtype=tp.float32)

        output = a @ b
        assert np.array_equal(cp.from_dlpack(output).get(), cp.from_dlpack(a).get() @ cp.from_dlpack(b).get())

    .. code-block:: python
        :linenos:
        :caption: Matrix-Vector Multiplication

        a = tp.iota((3,), dtype=tp.float32)
        b = tp.iota((3, 2), dtype=tp.float32)

        output = a @ b
        assert np.array_equal(cp.from_dlpack(output).get(), cp.from_dlpack(a).get() @ cp.from_dlpack(b).get())

    .. code-block:: python
        :linenos:
        :caption: Matrix-Matrix Multiplication

        a = tp.iota((2, 3), dtype=tp.float32)
        b = tp.iota((3, 2), dtype=tp.float32)

        output = a @ b
        assert np.array_equal(cp.from_dlpack(output).get(), cp.from_dlpack(a).get() @ cp.from_dlpack(b).get())

    .. code-block:: python
        :linenos:
        :caption: Batched Matrix Multiplication

        a = tp.iota((1, 2, 2, 2), dtype=tp.float32, dim=-1)
        b = tp.iota((1, 2, 2), dtype=tp.float32, dim=-2)

        output = a @ b
        assert np.array_equal(cp.from_dlpack(output).get(), cp.from_dlpack(a).get() @ cp.from_dlpack(b).get())
    """
    if self.rank == 0 or other.rank == 0:
        raise_error(
            "Input tensors must have at least 1 dimension.",
            [f"Note: Input tensors had {self.rank} and {other.rank} dimension(s) respectively."],
        )

    # Don't expand ranks if one of the inputs is rank 1:
    if self.rank != 1 and other.rank != 1:
        self, other = op_utils.match_ranks(self, other)
    return op_utils.create_op(MatrixMultiply, [self, other])
