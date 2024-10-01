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

from typing import Any, Sequence

from tripy import export, utils
from tripy.frontend.tensor import Tensor
from tripy.utils import Result


@export.public_api(document_under="modules", autodoc_options=[":no-members:", ":no-special-members:"])
class Parameter(Tensor):
    """
    A Parameter is a special kind of :class:`tripy.Tensor` that is treated by the compiler as a
    constant, enabling additional optimization opportunities.
    """

    def __init__(self, tensor: Any) -> None:
        """
        Args:
            tensor:
                The tensor value for this parameter. If provided as an external data format (e.g., a Numpy array),
                it will be converted into a Tripy Tensor.

        .. code-block:: python
            :linenos:
            :caption: Example

            parameter = tp.Parameter(tp.Tensor([1.0, 1.0], dtype=tp.float32))

            assert isinstance(parameter, tp.Parameter)
            assert isinstance(parameter, tp.Tensor)
        """
        t = tensor
        # for convenience, this will convert other dlpack-supporting representations too
        if not isinstance(t, Tensor):
            t = Tensor(t)
        self.__dict__ = t.__dict__

    def _is_compatible_helper(self, original_shape, other_shape, original_dtype, other_dtype) -> Result:
        if original_shape != other_shape:
            return Result.err(
                ["New parameter shape: ", other_shape, " is not compatible with current shape: ", original_shape]
            )
        if original_dtype != other_dtype:
            return Result.err(
                ["New parameter dtype: ", other_dtype, " is not compatible with current dtype: ", original_dtype]
            )
        return Result.ok()

    def _is_compatible(self, other: "Parameter") -> Result:
        # Determines whether another parameter has the same shape and
        # data type as this one.
        return self._is_compatible_helper(self.shape.tolist(), other.shape.tolist(), self.dtype, other.dtype)


class DefaultParameter(Parameter):
    """
    Behaves exactly like a parameter except does not cause
    data to be materialized for shape/dtype checks.
    Useful for initializing module parameters.
    """

    def __init__(self, shape: Sequence[int], dtype: "tripy.dtype") -> None:
        from tripy.frontend.ops.tensor_initializers import arange
        from tripy.frontend.trace.ops.reshape import reshape

        super().__init__(reshape(arange(utils.volume(shape), dtype), shape))

        self._shape = shape
        self._dtype = dtype

    def _is_compatible(self, other: "Parameter") -> Result:
        return self._is_compatible_helper(tuple(self._shape), tuple(other.shape.tolist()), self._dtype, other.dtype)
