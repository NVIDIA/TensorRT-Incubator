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
from typing import Sequence

from tripy import export, constraints
from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy.frontend.trace.ops import utils as op_utils


@dataclass(repr=False)
class Permute(BaseTraceOp):
    permutation: Sequence[int]

    # note that permuting a shape would not do anything
    infer_shape_output_idxs = op_utils.ShapeOutputIdxPolicies.infer_from_first_input_only

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import TransposeOp

        TransposeOp.build(inputs, outputs, perm=self.permutation)


@dataclass(repr=False)
class Transpose(Permute):
    """
    Represents a transpose operation.
    """

    dim0: int
    dim1: int

    def infer_rank(self):
        self.outputs[0].rank = self.inputs[0].rank
        perm = list(range(self.inputs[0].rank))
        perm[self.dim0], perm[self.dim1] = perm[self.dim1], perm[self.dim0]
        self.permutation = perm


@export.public_api(document_under="tensor_operations")
@constraints.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "float8", "int8", "int32", "int64", "bool"]},
    dtype_constraints={"input": "T1", constraints.RETURN_VALUE: "T1"},
)
def transpose(input: "tripy.Tensor", dim0: int, dim1: int) -> "tripy.Tensor":
    """
    Returns a new tensor that is a transposed version of the input tensor where
    ``dim0`` and ``dim1`` are swapped.

    Args:
        input: The input tensor.
        dim0: The first dimension to be transposed.
        dim1: The second dimension to be transposed.

    Returns:
        A new tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (2, 3))
        output = tp.transpose(input, 0, 1)

        assert np.array_equal(cp.from_dlpack(output).get(), np.transpose(np.arange(6, dtype=np.float32).reshape(2, 3), (1, 0)))
    """
    return Transpose.build([input], None, dim0, dim1)


@export.public_api(document_under="tensor_operations")
@constraints.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "float8", "int8", "int32", "int64", "bool"]},
    dtype_constraints={"input": "T1", constraints.RETURN_VALUE: "T1"},
)
def permute(input: "tripy.Tensor", perm: Sequence[int]) -> "tripy.Tensor":
    """
    Returns a tensor with its dimensions permuted.

    Args:
        input: The input tensor.
        perm: The desired ordering of dimensions.
              It must contain all integers in :math:`[0..N-1]` exactly once,
              where :math:`N` is the rank of the input tensor.

    Returns:
        A new tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (2, 3))
        output = tp.permute(input, (1, 0))

        assert np.array_equal(cp.from_dlpack(output).get(), np.transpose(np.arange(6, dtype=np.float32).reshape(2, 3), (1, 0)))
    """
    return Permute.build([input], perm)
