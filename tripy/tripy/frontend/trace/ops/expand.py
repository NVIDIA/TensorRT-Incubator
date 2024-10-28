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
from typing import Optional

from tripy import constraints, export
from tripy.common.exception import raise_error
from tripy.frontend import utils as frontend_utils
from tripy.frontend.trace.ops import utils as op_utils
from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy.types import ShapeLike


@dataclass(repr=False)
class Expand(BaseTraceOp):
    output_rank: int
    output_len: Optional[int] = None  # only used to help with infer_len for a shape input

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def infer_len(self):
        if self.output_len is not None:
            return [self.output_len]
        # if we don't have a static output length, we can't infer without evaluating the input
        return [None]

    def infer_rank(self):
        if self.output_rank is None:
            out_shape = op_utils.get_trace_shape(self.inputs[1])
            assert len(out_shape) == 1
            assert out_shape[0] >= 0, f"incorrect shape computation {out_shape}"
            self.output_rank = out_shape[0]

        self.outputs[0].rank = self.output_rank

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import DynamicBroadcastOp

        broadcast_dim = op_utils.get_broadcast_in_dim(inputs[0].rank, outputs[0].rank)

        DynamicBroadcastOp.build(
            [inputs[0], inputs[1]],
            outputs,
            broadcast_dim=broadcast_dim,
        )


@export.public_api(document_under="operations/functions")
@constraints.dtypes(
    constraints={"input": "T1", constraints.RETURN_VALUE: "T1"},
    variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int8", "int32", "int64", "bool"],
    },
)
def expand(input: "tripy.Tensor", sizes: ShapeLike) -> "tripy.Tensor":
    """
    Returns a new tensor based on the input tensor with singleton dimensions expanded to a larger size.

    Args:
        input: The input tensor.
        sizes: The desired expanded size.
            A value of :math:`-1` indicates that the dimension should not be modified.
            If the length of this parameter exceeds the rank of the tensor, new dimensions
            are prepended.

    Returns:
        The new tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.iota((2, 1), dtype=tp.float32)
        output = tp.expand(input, (-1, 4))

        assert np.array_equal(cp.from_dlpack(output).get(), np.broadcast_to(cp.from_dlpack(input).get(), (2, 4)))

    .. code-block:: python
        :linenos:
        :caption: Increasing Tensor Rank

        input = tp.iota((1, 1), dtype=tp.float32)
        output = tp.expand(input, (3, -1, -1))

        assert np.array_equal(cp.from_dlpack(output).get(), np.broadcast_to(cp.from_dlpack(input).get(), (3, 1, 1)))
    """

    if len(sizes) < input.rank:
        raise_error(
            "The length of `sizes` must be greater or equal to input tensor's rank.",
            [f"sizes has length: {len(sizes)}", f" input rank: {input.rank}"],
        )

    num_prepended = len(sizes) - input.rank
    out_shape = list(sizes[:num_prepended]) + [
        inp_dim if op_utils.is_minus_one(out_dim) else out_dim
        for inp_dim, out_dim in zip(input.shape, sizes[num_prepended:])
    ]

    if any(op_utils.is_minus_one(dim) for dim in out_shape):
        raise_error(
            "Cannot use -1 for prepended dimension.",
            [
                f"{num_prepended} dimension(s) are going to be prepended since the `sizes` argument "
                f"contains more elements than the number of dimensions in the input.\n"
                f"Prepended dimensions may not contain -1 since there is no corresponding "
                f"dimension in the input to copy from, but got: {sizes}"
            ],
        )

    # only used for inferring the length of a shape output (hence, define only in rank-1 case)
    out_len = None
    if len(sizes) == 1 and isinstance(out_shape[0], int):
        out_len = out_shape[0]

    # TODO (pranavm): Remove output_rank/output_len here and elsewhere.
    # We should be able to get the shape of the trace tensor for `shape` one infer_dynamic_shapes is implemented.
    @frontend_utils.convert_to_tensors(skip_num_stack_entries=1)
    def expand_impl(input: "tripy.Tensor", shape: ShapeLike, output_rank: int, output_len: Optional[int] = None):
        return Expand.build([input, shape], output_rank, output_len)

    return expand_impl(input, out_shape, output_rank=len(sizes), output_len=out_len)
