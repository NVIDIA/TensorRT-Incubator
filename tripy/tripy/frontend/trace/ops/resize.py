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
from tripy.frontend import utils as frontend_utils
from tripy.frontend.trace.ops import utils as op_utils
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Resize(BaseTraceOp):

    mode: str
    scales: Sequence[float]
    align_corners: bool

    infer_tensor_variants = op_utils.InferVariantPolicies.never_return_shape

    def infer_rank(self):
        self.outputs[0].rank = self.inputs[0].rank

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ResizeCubicOp, ResizeLinearOp, ResizeNearestOp

        if self.scales:
            from tripy.common.datatype import float32, int32
            from tripy.flat_ir.ops import ConstantOp, ConvertOp, MulOp
            from tripy.flat_ir.tensor import FlatIRTensor

            # construct output_shape using scales
            # inputs[1] is input[0].shape
            # output_shape = (inputs[1].cast(fp32) * scales).cast(int32)
            out_shape = (inputs[0].rank,)
            scales_tensor = FlatIRTensor.build(
                shape=out_shape,
                rank=1,
                dtype=float32,
                device=outputs[0].device,
                reason_details=[f"create scales tensor in resize op."],
            )
            ConstantOp.build([], [scales_tensor], data=self.scales)
            input_shape_f32 = FlatIRTensor.build(
                shape=out_shape,
                rank=1,
                dtype=float32,
                device=outputs[0].device,
                reason_details=[f"convert input shape tensor to float32 in resize op."],
            )
            ConvertOp.build([inputs[1]], [input_shape_f32])
            out_shape_f32 = FlatIRTensor.build(
                shape=out_shape,
                rank=1,
                dtype=float32,
                device=outputs[0].device,
                reason_details=[f"compute output shape in resize op."],
            )
            MulOp.build([input_shape_f32, scales_tensor], [out_shape_f32])
            out_shape_tensor = FlatIRTensor.build(
                shape=out_shape,
                rank=1,
                dtype=int32,
                device=outputs[0].device,
                reason_details=[f"convert output shape to int32 in resize op."],
            )
            ConvertOp.build([out_shape_f32], [out_shape_tensor])
            inputs[1] = out_shape_tensor

        if self.mode == "nearest":
            ResizeNearestOp.build(inputs, outputs)
        elif self.mode == "cubic":
            ResizeCubicOp.build(inputs, outputs, self.align_corners, cubic_coeff=-0.75)
        else:
            ResizeLinearOp.build(inputs, outputs, self.align_corners)


@export.public_api(document_under="operations/functions")
@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "int8"],
    },
    dtype_constraints={"input": "T1", constraints.RETURN_VALUE: "T1"},
)
@frontend_utils.convert_shape_inputs(["output_shape"])
def resize(
    input: "tripy.Tensor",
    mode: str,
    output_shape: "tripy.types.ShapeLike" = None,
    scales: Sequence[Union[int, float]] = None,
    align_corners: bool = False,
) -> "tripy.Tensor":
    r"""
    Resizes the input tensor.

    Args:
        input: The input tensor.
        mode: The resize operation's algorithm. Must be one of: ["cubic", linear", "nearest"].
        output_shape: The output shape of the resize operation.
        scales: A sequence of scale factors for each dimension. Must have
            the same length as input tensor's rank. Will be ignored if ``output_shape`` is
            given.
        align_corners: If set to ``True``, the input and output tensors are aligned by the
            center points of their corner pixels, preserving the values at the corner pixels.
            If set to ``False``, the input and output tensors are aligned by the corner points of
            their corner pixels. Only in effect when ``mode`` is ``"cubic"`` or ``"linear"``.

    Returns:
        The output tensor after the resize operation.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.reshape(tp.arange(16, dtype=tp.float32), (1, 1, 4, 4))
        output = tp.resize(input, "nearest", scales=(1, 1, 2, 2))

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
    if align_corners and mode not in ("cubic", "linear"):
        raise_error("align_corners can only be set with `cubic` or `linear` mode.")
    if output_shape is None and scales is None:
        raise_error("One of `output_shape` and `scale` must be given.")

    inputs = [input]
    inputs.append(output_shape if output_shape is not None else input.shape)
    return Resize.build(inputs, mode, scales, align_corners)
