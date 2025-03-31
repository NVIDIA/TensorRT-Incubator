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

import numbers
from typing import Sequence

from nvtripy import export
from nvtripy.common.exception import raise_error
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.trace.ops.resize import ResizeCubic, ResizeLinear, ResizeNearest
from nvtripy.types import ShapeLike
from nvtripy.utils import wrappers


SUPPORTED_MODES = ("cubic", "linear", "nearest")


def _check_mode(mode: str, align_corners: bool):
    if mode not in SUPPORTED_MODES:
        raise_error(
            "Unsupported resize mode.",
            [f"Supported modes are {SUPPORTED_MODES}, but got {mode}."],
        )
    if align_corners and mode not in ("cubic", "linear"):
        raise_error("align_corners can only be set with `cubic` or `linear` mode.")


def _create_resize(mode, inputs, scales, align_corners):
    if mode == "nearest":
        return op_utils.create_op(ResizeNearest, inputs, scales=scales)
    elif mode == "linear":
        return op_utils.create_op(ResizeLinear, inputs, scales=scales, align_corners=align_corners)
    else:
        return op_utils.create_op(ResizeCubic, inputs, scales=scales, align_corners=align_corners)


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "int8"]},
    convert_to_tensors=True,
)
def resize(
    input: "nvtripy.Tensor", output_shape: ShapeLike, mode: str = "linear", align_corners: bool = False
) -> "nvtripy.Tensor":
    r"""
    Resizes the input tensor.

    Args:
        input: The input tensor.
        mode: The resize operation's algorithm. Must be one of: ["cubic", linear", "nearest"].
        output_shape: The output shape of the resize operation.
        align_corners: If set to ``True``, the input and output tensors are aligned by the
            center points of their corner pixels, preserving the values at the corner pixels.
            If set to ``False``, the input and output tensors are aligned by the corner points of
            their corner pixels. Only in effect when ``mode`` is ``"cubic"`` or ``"linear"``.

    Returns:
        The output tensor after the resize operation.

    .. code-block:: python
        :linenos:
        :caption: Nearest Neighbor Interpolation

        input = tp.reshape(tp.arange(4), (1, 1, 2, 2))
        output = tp.resize(input, output_shape=(1, 1, 4, 4), mode="nearest")

        expected = torch.nn.functional.interpolate(torch.from_dlpack(input), scale_factor=2.0, mode="nearest") # doc: omit
        assert torch.allclose(torch.from_dlpack(output), expected)

    .. code-block:: python
        :linenos:
        :caption: Linear Interpolation

        input = tp.reshape(tp.arange(4), (1, 1, 2, 2))
        output = tp.resize(input, output_shape=(1, 1, 4, 4), mode="linear")

        expected = torch.nn.functional.interpolate(torch.from_dlpack(input), scale_factor=2.0, mode="bilinear") # doc: omit
        assert torch.allclose(torch.from_dlpack(output), expected)

    .. code-block:: python
        :linenos:
        :caption: Cubic Interpolation

        input = tp.reshape(tp.arange(4), (1, 1, 2, 2))
        output = tp.resize(input, output_shape=(1, 1, 4, 4), mode="cubic")

        expected = torch.nn.functional.interpolate(torch.from_dlpack(input), scale_factor=2.0, mode="bicubic") # doc: omit
        assert torch.allclose(torch.from_dlpack(output), expected)
    """
    _check_mode(mode, align_corners)
    return _create_resize(mode, [input, output_shape], scales=None, align_corners=align_corners)


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "int8"]},
)
def resize(
    input: "nvtripy.Tensor", scales: Sequence[numbers.Number], mode: str = "linear", align_corners: bool = False
) -> "nvtripy.Tensor":
    r"""
    Resizes the input tensor.

    Args:
        input: The input tensor.
        mode: The resize operation's algorithm. Must be one of: ["cubic", linear", "nearest"].
        scales: A sequence of scale factors for each dimension. Must have
            the same length as input tensor's rank.
        align_corners: If set to ``True``, the input and output tensors are aligned by the
            center points of their corner pixels, preserving the values at the corner pixels.
            If set to ``False``, the input and output tensors are aligned by the corner points of
            their corner pixels. Only in effect when ``mode`` is ``"cubic"`` or ``"linear"``.

    Returns:
        The output tensor after the resize operation.

    .. code-block:: python
        :linenos:
        :caption: Nearest Neighbor Interpolation

        input = tp.reshape(tp.arange(4), (1, 1, 2, 2))
        output = tp.resize(input, scales=(1, 1, 2, 2), mode="nearest")

        expected = torch.nn.functional.interpolate(torch.from_dlpack(input), scale_factor=2.0, mode="nearest") # doc: omit
        assert torch.allclose(torch.from_dlpack(output), expected)

    .. code-block:: python
        :linenos:
        :caption: Linear Interpolation

        input = tp.reshape(tp.arange(4), (1, 1, 2, 2))
        output = tp.resize(input, scales=(1, 1, 2, 2), mode="linear")

        expected = torch.nn.functional.interpolate(torch.from_dlpack(input), scale_factor=2.0, mode="bilinear") # doc: omit
        assert torch.allclose(torch.from_dlpack(output), expected)

    .. code-block:: python
        :linenos:
        :caption: Cubic Interpolation

        input = tp.reshape(tp.arange(4), (1, 1, 2, 2))
        output = tp.resize(input, scales=(1, 1, 2, 2), mode="cubic")

        expected = torch.nn.functional.interpolate(torch.from_dlpack(input), scale_factor=2.0, mode="bicubic") # doc: omit
        assert torch.allclose(torch.from_dlpack(output), expected)
    """
    _check_mode(mode, align_corners)

    return _create_resize(mode, [input], scales=scales, align_corners=align_corners)
