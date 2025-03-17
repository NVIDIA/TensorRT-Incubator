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

from typing import Sequence, Tuple, Union

from nvtripy import export
from nvtripy.common.exception import raise_error
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.trace.ops.shape import Shape
from nvtripy.trace.ops.slice import SliceFill
from nvtripy.types import IntLike
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bool", "int32", "int64"]},
)
def pad(
    input: "nvtripy.Tensor",
    pad: Sequence[Tuple[IntLike, IntLike]],
    mode: str = "constant",
    value: Union[int, float] = 0,
) -> "nvtripy.Tensor":
    r"""
    Pads the input tensor.

    Args:
        input: The input tensor.
        pad: A sequence of padding sizes of each dimension. Its length must be equal to the rank
            of ``input``. Each element of ``pad`` is a tuple of integers or :class:`DimensionSize` s ``(low, high)``,
            which represents the padding sizes before the lowest index and after the highest index at
            the corresponding dimension.
        mode: The padding mode. Only "constant" is supported.
        value: The padding value for "constant" mode.

    Returns:
        The padded tensor.

    .. code-block:: python
        :linenos:
        :caption: Constant padding.

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (2, 3))
        output = tp.pad(input, [(1, 0), (0, 1)])

        input_np = np.arange(6, dtype=np.float32).reshape((2, 3)) # doc: omit
        expected = np.pad(input_np, ((1, 0), (0, 1))) # doc: omit
        assert np.array_equal(cp.from_dlpack(output).get(), expected)
    """
    from nvtripy.frontend.tensor import Tensor

    if len(pad) != input.rank:
        raise_error(
            "`pad` length must equal to the rank of `input`.",
            [f"Got pad={pad}, ", f" input's rank={input.rank}"],
        )

    supported_modes = {"constant"}
    if mode not in supported_modes:
        raise_error(
            "Unsupported padding mode.",
            [f"Got mode={mode}, while supported modes are {supported_modes}"],
        )

    padding_lows, padding_highs = list(zip(*pad))
    padding_lows = op_utils.tensor_from_shape_like(padding_lows)
    padding_highs = op_utils.tensor_from_shape_like(padding_highs)
    starts = -padding_lows

    # Not using input.shape because we need a `Tensor` here
    input_shape = op_utils.create_op(Shape, [input])
    sizes = input_shape + padding_lows + padding_highs
    steps = op_utils.tensor_from_shape_like([1] * input.rank)

    return op_utils.create_op(SliceFill, [input, starts, sizes, steps, Tensor(value, dtype=input.dtype)])
