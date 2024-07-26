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

from tripy import export, utils
from tripy.utils import Result
from tripy.common.exception import raise_error
from tripy.frontend.trace.ops.base import BaseTraceOp
import tripy.frontend.trace.ops.utils as op_utils


@dataclass(repr=False)
class Expand(BaseTraceOp):
    shape: Sequence[int]

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def infer_shape_output_idxs(self, inputs) -> Result:
        from tripy.frontend.shape import Shape

        # wrap if the first input is a shape and the output is rank-1
        if isinstance(inputs[0], Shape) and (
            (self.shape and len(self.shape) == 1) or self.get_shape_input_length() == 1
        ):
            return Result.ok([0])
        return Result.ok([])

    def infer_rank(self):
        if self.shape:
            if len(self.shape) < self.inputs[0].rank:
                utils.raise_error_io_info(
                    self,
                    "The number of sizes must be greater or equal to input tensor's rank.",
                    [f"Target sizes: {self.shape}", f" input rank: {self.inputs[0].rank}"],
                )
            self.outputs[0].rank = len(self.shape)
        else:
            output_rank = self.get_shape_input_length()
            self.outputs[0].rank = output_rank

            if output_rank < self.inputs[0].rank:
                utils.raise_error_io_info(
                    self,
                    "The shape of size tensor must be greater or equal to input tensor's rank.",
                    [f"Target sizes shape: {output_rank}", f" input rank: {self.inputs[0].rank}"],
                )

    def get_shape_input_length(self):
        from tripy.backend.mlir.utils import ShapeContext

        out_shape = ShapeContext().get_shape_of_dynamic_trace_tensor(self.inputs[1])
        assert len(out_shape) == 1, f"Rank of sizes tensor is expected to be 1, got {len(out_shape)}."
        return out_shape[0]

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import DynamicBroadcastOp

        broadcast_dim = op_utils.get_broadcast_in_dim(inputs[0].rank, outputs[0].rank)

        output_shape = (
            op_utils.add_constant_tensor_from_list(self.shape, device=inputs[0].device)
            if len(inputs) == 1
            else inputs[1]
        )

        DynamicBroadcastOp.build(
            [inputs[0], output_shape],
            outputs,
            broadcast_dim=broadcast_dim,
        )


@export.public_api(document_under="tensor_operations")
def expand(input: "tripy.Tensor", sizes: Union[Sequence[int], "tripy.Tensor"]) -> "tripy.Tensor":
    """
    Returns a new tensor based on the input tensor with singleton dimensions expanded to a larger size.

    Args:
        input: The input tensor.
        sizes: The desired expanded size.
            A value of :math:`-1` indicates that the dimension should not be modified.
            If the length of this parameter exceeds the rank of the tensor, new dimensions
            are prepended.

    Returns:
        The new tensor of the same data type as this tensor.

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
    from tripy.frontend.trace.ops.concatenate import concatenate
    from tripy.frontend.trace.ops.reshape import reshape

    if isinstance(sizes, Tensor):
        if sizes.rank != 1:
            raise_error(
                "Expand operation requires sizes tensor to be of rank 1.",
                [
                    f"sizes tensor must of rank 1,",
                    f"Got rank={sizes.rank}",
                ],
            )
        return Expand.build([input, sizes], None)

    args = []
    for i, idx in enumerate(sizes):
        t_shape = input.shape

        def convert_to_positive_idx(index: int) -> Tensor:
            # Base condition for t_shape[i] else the frontend will recurse infinitely.
            assert isinstance(index, int)
            return Tensor([index]) if index >= 0 else reshape(index + t_shape[i] + 1, (1,))

        args.append(convert_to_positive_idx(idx) if i < input.rank else Tensor([1]))

    sizes = concatenate(args, dim=0)
    return Expand.build([input, sizes], None)
