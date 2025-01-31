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

from nvtripy.common.exception import raise_error
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.frontend.ops._registry import register_tensor_method
from nvtripy.trace.ops.matmul import MatrixMultiplication
from nvtripy.utils import wrappers


@register_tensor_method("__matmul__")
@wrappers.interface(
    dtype_constraints={"self": "T1", "other": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int32"]},
)
def __matmul__(self: "nvtripy.Tensor", other: "nvtripy.Tensor") -> "nvtripy.Tensor":
    """
    Performs matrix multiplication between two tensors.

    - If both tensors are 1D, a dot product is performed.
    - If both tensors are 2D, matrix multiplication is performed.
    - If either argument, but not both, is 1D, matrix-vector multiplication is performed.
    - If both tensors are 2D or higher dimensional and have differnt ranks, a dimension is inserted
        and batched matrix multiplication is performed with broadcast of relevant dimension.

    Args:
        self: Tensor to be multiplied with other.
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
    lhs_rank = self.rank
    rhs_rank = other.rank

    for index, rank in enumerate([lhs_rank, rhs_rank]):
        if rank < 1:
            raise_error(
                "Input tensors must have at least 1 dimension.",
                details=[
                    f"Inputs for operation: '@' must have at least one dimension, but input {index} has rank: {rank}."
                ],
            )

    if lhs_rank == 1 and rhs_rank == 1:
        # case 1: both operands are 1-D
        batching_dim = {"lhs": [], "rhs": []}
        contracting_dim = {"lhs": [0], "rhs": [0]}
    else:
        # stablehlo dot_general requires same number of batching dims for lhs, rhs.

        def compute_contracting_dims(rank_a, rank_b):
            is_vector = lambda rank: rank == 1
            is_matrix = lambda rank: rank >= 2

            if is_vector(rank_a) and is_vector(rank_b):
                # Vector-vector multiplication
                return [[0], [0]]
            elif is_vector(rank_a) and is_matrix(rank_b):
                # Vector-matrix multiplication
                return [[0], [rank_b - 2]]
            elif is_matrix(rank_a) and is_vector(rank_b):
                # Matrix-vector multiplication
                return [[rank_a - 1], [0]]
            else:
                # Matrix-matrix multiplication (or higher-rank tensor multiplication)
                output_rank = max(rank_a, rank_b)
                return [[output_rank - 1], [output_rank - 2]]

        def get_batch_indices(rank):
            return list(range(rank - 2))

        output_rank = 1 if lhs_rank == 1 or rhs_rank == 1 else max(lhs_rank, rhs_rank)
        batching_dim = {"lhs": get_batch_indices(output_rank), "rhs": get_batch_indices(output_rank)}
        contracting_dim = compute_contracting_dims(lhs_rank, rhs_rank)
        contracting_dim = {
            "lhs": contracting_dim[0],
            "rhs": contracting_dim[1],
        }

    return op_utils.create_op(MatrixMultiplication, [self, other], contracting_dim, batching_dim)
