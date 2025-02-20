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

from typing import Sequence, Union

from nvtripy import export
from nvtripy.common.exception import raise_error
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.trace.ops.pad import Pad
from nvtripy.types import ShapeLike
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bool", "int32"]},
)
def pad(
    input: "nvtripy.Tensor", pad: Sequence[ShapeLike], mode: str = "constant", value: Union[int, float] = 0
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
        output = tp.pad(input, ((1, 0), (0, 1)))

        input_np = np.arange(6, dtype=np.float32).reshape((2, 3)) # doc: omit
        expected = np.pad(input_np, ((1, 0), (0, 1))) # doc: omit
        assert np.array_equal(cp.from_dlpack(output).get(), expected)
    """
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

    padding_low, padding_high = list(zip(*pad))
    return op_utils.create_op(
        Pad,
        [
            input,
            op_utils.tensor_from_shape_like(padding_low),
            op_utils.tensor_from_shape_like(padding_high),
        ],
        value,
    )
