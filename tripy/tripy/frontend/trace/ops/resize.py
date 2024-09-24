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
from typing import Sequence, Union

from tripy import export, constraints
from tripy.common.exception import raise_error
from tripy.frontend.trace.ops import utils as op_utils
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Resize(BaseTraceOp):

    mode: str
    scales: Sequence[float]
    align_corners: bool

    infer_shape_output_idxs = op_utils.ShapeOutputIdxPolicies.never_return_shape

    def infer_rank(self):
        self.outputs[0].rank = self.inputs[0].rank

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ResizeCubicOp, ResizeLinearOp, ResizeNearestOp

        if self.mode == "nearest":
            ResizeNearestOp.build(inputs, outputs, self.scales)
        elif self.mode == "cubic":
            ResizeCubicOp.build(inputs, outputs, self.scales, self.align_corners)
        else:
            ResizeLinearOp.build(inputs, outputs, self.scales, self.align_corners)


@export.public_api(document_under="operations/functions")
@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "int8"],
    },
    dtype_constraints={"input": "T1", constraints.RETURN_VALUE: "T1"},
)
def resize(
    input: "tripy.Tensor",
    mode: str,
    scales: Sequence[Union[int, float]],
    align_corners: bool = False,
) -> "tripy.Tensor":
    r"""
    Resizes the input tensor.

    Args:
        input: The input tensor.
        mode: The resize operation's algorithm. Must be one of "cubic, linear, nearest".
        scales: A sequence of scale factors for each dimension. Must have
            the same length as input tensor's rank.
        align_corners: If set to ``True``, the input and output tensors are aligned by the
            center points of their corner pixels, preserving the values at the corner pixels.
            If set to ``False``, the input and output tensors are aligned by the corner points of
            their corner pixels. Only in effect when ``mode`` is ``cubic`` or ``linear``.

    Returns:
        The output tensor after the resize operation.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.reshape(tp.arange(16, dtype=tp.float32), (1, 1, 4, 4))
        output = tp.interpolate(input, "nearest", scales=(1, 1, 2, 2))

        input_torch = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4)) # doc: omit
        expected = torch.nn.functional.interpolate(input_torch, scale_factor=2.0, mode="nearest") # doc: omit

        assert torch.allclose(torch.from_dlpack(output).to("cpu"), expected)
    """
    supported_modes = ("cubic", "linear", "nearest")
    if mode not in supported_modes:
        raise_error(
            "Unsupported resize mode.",
            [f"Supported modes are {supported_modes}, but got {mode}."],
        )
    return Resize.build([input], mode, scales, align_corners)
