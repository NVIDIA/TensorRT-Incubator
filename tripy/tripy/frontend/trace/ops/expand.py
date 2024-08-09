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
from typing import Sequence, Union

from tripy import export, constraints
from tripy.utils import Result
from tripy.common.exception import raise_error
from tripy.frontend import utils as frontend_utils
from tripy.frontend.trace.ops import utils as op_utils
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Expand(BaseTraceOp):
    output_rank: int

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def infer_shape_output_idxs(self, inputs) -> Result:
        from tripy.frontend.shape import Shape

        # wrap if the first input is a shape and the output is rank-1
        if isinstance(inputs[0], Shape) and self.output_rank == 1:
            return Result.ok([0])
        return Result.ok([])

    def infer_rank(self):
        if self.output_rank is None:
            from tripy.backend.mlir.utils import ShapeContext

            out_shape = ShapeContext().get_shape_of_dynamic_trace_tensor(self.inputs[1])
            assert len(out_shape) == 1
            assert out_shape[0] >= 0, f"incorrect shape computation {out_shape}"
            self.inputs[1].shape = out_shape
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


@frontend_utils.convert_inputs_to_tensors(exclude=["input", "output_rank"], shape_argument=["shape"])
def expand_impl(input: "tripy.Tensor", shape: Sequence, output_rank: int):
    return Expand.build([input, shape], output_rank)


@export.public_api(document_under="tensor_operations")
@constraints.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64", "bool"], "T2": [ "int8", "int32", "int64"]},
    dtype_constraints={"input": "T1", "sizes": "T2", constraints.RETURN_VALUE: "T1"},
)
def expand(input: "tripy.Tensor", sizes: Union["tripy.Shape", Sequence[Union[int, "tripy.Tensor"]]]) -> "tripy.Tensor":
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
    from tripy.frontend.tensor import Tensor

    if isinstance(sizes, Tensor):
        return Expand.build([input, sizes], None)

    if len(sizes) < input.rank:
        raise_error(
            "The length of `sizes` must be greater or equal to input tensor's rank.",
            [f"sizes has length: {len(sizes)}", f" input rank: {input.rank}"],
        )

    idx_offset = len(sizes) - input.rank
    out_shape = []
    for i, size in enumerate(sizes):
        if isinstance(size, int) and size == -1:
            # keep the original dimension
            out_shape.append(input.shape[i - idx_offset])
            continue
        out_shape.append(size)

    return expand_impl(input, out_shape, len(sizes))
