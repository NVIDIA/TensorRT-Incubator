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
from typing import Sequence, Tuple, Union

from nvtripy import export
from nvtripy.common.shape_bounds import ShapeBounds, ValueBounds
from nvtripy.frontend.dimension_size import DimensionSize
from nvtripy.types import IntLike


@export.public_api(document_under="compiling_code")
class InputInfo:
    """
    Captures information about an input to a compiled function.
    """

    def __init__(
        self, shape: Sequence[Union[IntLike, Tuple[IntLike, IntLike, IntLike]]], dtype: "nvtripy.dtype"
    ) -> None:
        """
        Args:
            shape: The shape of the input.
                To indicate dynamic dimensions, provide the minimum, optimum, and maximum values for the dimension.
            dtype: The data type of the input.

        .. code-block:: python
            :linenos:

            inp = tp.InputInfo((2, 4), dtype=tp.float32)
            assert inp.shape_bounds.min == [2, 4]
            assert inp.shape_bounds.opt == [2, 4]
            assert inp.shape_bounds.max == [2, 4]

        .. code-block:: python
            :linenos:
            :caption: Dynamic Dimensions

            # The first dimension will support values in the range [1, 3],
            # optimizing for a size of 2.
            inp = tp.InputInfo(((1, 2, 3), 4), dtype=tp.float32)
            assert inp.shape_bounds.min == [1, 4]
            assert inp.shape_bounds.opt == [2, 4]
            assert inp.shape_bounds.max == [3, 4]
        """
        is_int_like = lambda arg: any(isinstance(arg, typ) for typ in {int, DimensionSize})

        min_shape = []
        opt_shape = []
        max_shape = []
        for elem in shape:
            if is_int_like(elem):
                elem = (elem,) * 3

            assert len(elem) == 3 and all(is_int_like(val) for val in elem)

            min_shape.append(elem[0])
            opt_shape.append(elem[1])
            max_shape.append(elem[2])

        self.shape_bounds = ShapeBounds(min_shape, opt_shape, max_shape)
        self.dtype = dtype

    def __str__(self) -> str:
        return f"InputInfo(min={self.shape_bounds.min}, opt={self.shape_bounds.opt}, max={self.shape_bounds.max}, dtype={self.dtype})"


@export.public_api(document_under="compiling_code")
class DimensionInputInfo:
    """
    Captures information about a dimension size input to a compiled function.
    """

    def __init__(self, value_bounds: Tuple[IntLike, IntLike, IntLike]) -> None:
        """
        Args:
            value_bounds: The value bound of the dimension size input, consisting of minimum, optimum, and maximum values.

        .. code-block:: python
            :linenos:
            :caption: Dynamic Dimensions

            # The dimension size will support values in the range [1, 3],
            # optimizing for a size of 2.
            dim_inp = tp.DimensionInputInfo((1, 2, 3))
            assert dim_inp.min == 1
            assert dim_inp.opt == 2
            assert dim_inp.max == 3
        """
        self.value_bounds = ValueBounds(min=[value_bounds[0]], opt=[value_bounds[1]], max=[value_bounds[2]])

    def __str__(self) -> str:
        return (
            f"DimensionInputInfo(min={self.value_bounds.min}, opt={self.value_bounds.opt}, max={self.value_bounds.max})"
        )
