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
from typing import Dict, Sequence, Tuple, Union

from nvtripy import export
from nvtripy.backend.api.named_dimension import NamedDimension
from nvtripy.backend.api.bounds import Bounds
from nvtripy.frontend.dimension_size import DimensionSize
from nvtripy.types import IntLike
from nvtripy.utils import json as json_utils


@export.public_api(document_under="compiling_code/input_info/index.rst")
class InputInfo:
    """
    Captures information about an input to a compiled function.
    """

    def __init__(
        self,
        shape: Sequence[Union[NamedDimension, IntLike, Tuple[IntLike, IntLike, IntLike]]],
        dtype: "nvtripy.dtype",
    ) -> None:
        """
        Args:
            shape: The shape of the input.
                To indicate dynamic dimensions, provide the minimum, optimum, and maximum values for the dimension.
            dtype: The data type of the input.

        .. code-block:: python
            :linenos:

            inp = tp.InputInfo((2, 4), dtype=tp.float32)
            assert inp.shape_bounds.min == (2, 4)
            assert inp.shape_bounds.opt == (2, 4)
            assert inp.shape_bounds.max == (2, 4)

        .. code-block:: python
            :linenos:
            :caption: Dynamic Dimensions

            # The first dimension will support values in the range [1, 3],
            # optimizing for a size of 2.
            inp = tp.InputInfo(((1, 2, 3), 4), dtype=tp.float32)
            assert inp.shape_bounds.min == (1, 4)
            assert inp.shape_bounds.opt == (2, 4)
            assert inp.shape_bounds.max == (3, 4)

        .. code-block:: python
            :linenos:
            :caption: Naming Dynamic Dimensions

            # Dimensions with the same name must be equal at runtime.
            # This knowledge can help the compiler optimize better.
            window_size = tp.NamedDimension("window_size", 3, 5, 7)

            inp = tp.InputInfo((1, window_size, window_size), dtype=tp.float32)
            assert inp.shape_bounds.min == (1, 3, 3)
            assert inp.shape_bounds.opt == (1, 5, 5)
            assert inp.shape_bounds.max == (1, 7, 7)
            assert inp.dimension_names == {1: "window_size", 2: "window_size"}
        """
        is_int_like = lambda arg: any(isinstance(arg, typ) for typ in {int, DimensionSize})

        min_shape = []
        opt_shape = []
        max_shape = []
        dimension_names = {}
        for idx, elem in enumerate(shape):
            if is_int_like(elem):
                elem = (elem,) * 3

            if isinstance(elem, NamedDimension):
                dimension_names[idx] = elem.name
                elem = elem.bounds

            assert len(elem) == 3 and all(is_int_like(val) for val in elem)

            min_shape.append(elem[0])
            opt_shape.append(elem[1])
            max_shape.append(elem[2])

        self.dimension_names: Dict[int, str] = dimension_names
        """
        A mapping of dimension indices to their names, if set.
        """

        self.shape_bounds: Bounds = Bounds(tuple(min_shape), tuple(opt_shape), tuple(max_shape))
        """
        The shape bounds of the input.
        """
        self.dtype: "nvtripy.dtype" = dtype
        """
        The data type of the input.
        """

    def __str__(self) -> str:
        return f"InputInfo<{self.shape_bounds}, dimension names: {self.dimension_names}, dtype: {self.dtype}>"

    def __eq__(self, other):
        return isinstance(other, InputInfo) and self.shape_bounds == other.shape_bounds and self.dtype == other.dtype


@json_utils.Encoder.register(InputInfo)
def encode_input_info(input_info):
    return {
        "shape_bounds": input_info.shape_bounds,
        "dimension_names": input_info.dimension_names,
        "dtype": input_info.dtype,
    }


@json_utils.Decoder.register(InputInfo)
def decode_input_info(input_info_dict):
    input_info = InputInfo(shape=[], dtype=input_info_dict["dtype"])
    input_info.shape_bounds = input_info_dict["shape_bounds"]
    input_info.dimension_names = {int(k): v for k, v in input_info_dict.get("dimension_names", {}).items()}
    return input_info


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
        # Evaluate `DimensionSize` early to avoid duplicate evaluation
        value_bounds = tuple(map(int, value_bounds))
        self.value_bounds = Bounds(
            min=tuple([value_bounds[0]]), opt=tuple([value_bounds[1]]), max=tuple([value_bounds[2]])
        )

    def __str__(self) -> str:
        return (
            f"DimensionInputInfo(min={self.value_bounds.min}, opt={self.value_bounds.opt}, max={self.value_bounds.max})"
        )


@json_utils.Encoder.register(DimensionInputInfo)
def encode_dim_input_info(dim_input_info):
    return {
        "value_bounds": dim_input_info.value_bounds,
    }


@json_utils.Decoder.register(DimensionInputInfo)
def decode_dim_input_info(dim_input_info_dict):
    dim_input_info = DimensionInputInfo((-1, -1, -1))
    dim_input_info.value_bounds = dim_input_info_dict["value_bounds"]
    return dim_input_info
