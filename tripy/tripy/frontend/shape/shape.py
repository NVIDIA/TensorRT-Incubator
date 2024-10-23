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

from textwrap import indent
from typing import Any, Optional, Sequence, Union

# Import ops to populate the registry before we define our Tensor class
import tripy.frontend.ops
import tripy.frontend.trace.ops
import tripy.frontend.utils as frontend_utils
from tripy import constraints, export, utils
from tripy.common.datatype import int32
from tripy.common.exception import raise_error
from tripy.frontend.shape.shape_scalar import ShapeScalar
from tripy.frontend.tensor import BaseTensor, Tensor
from tripy.frontend.trace.ops import Storage
from tripy.frontend.trace.tensor import TraceTensor
from tripy.utils.stack_info import StackInfo


# TODO (pranavm): Make Shape not a subclass of Tensor.
@export.public_api(document_under="shape/index.rst")
class Shape(BaseTensor):
    """
    Represents the shape of a :class:`Tensor` .

    :class:`Shape` s are intended to behave like lists of non-negative integers of :class:`int32` data type.
    """

    _COUNT = 0

    # set to higher than tensor's so adding a shape to a tensor will use shape's overload (and give an error)
    __array_priority__ = 20000

    @classmethod
    def _get_unique_name(cls):
        name = f"t{cls._COUNT}"
        cls._COUNT += 1
        return name

    def __init__(
        self,
        data: Sequence[Union[int, ShapeScalar]],
        name: Optional[str] = None,
        fetch_stack_info: bool = True,
    ) -> None:
        r"""
        Args:
            data: The value of the shape, which should be a 1D array of integers (the dimensions).
            name: The name of the shape. If provided, this must be a unique string.
            fetch_stack_info: Whether to fetch stack information for the shape.
                Stack information allows Tripy to generate much higher quality error
                messages at the cost of a small overhead when initializing the shape.

        """
        stack_info = StackInfo([])
        if fetch_stack_info:
            stack_info = utils.get_stack_info()

        name = name if name is not None else Tensor._get_unique_name()
        self.trace_tensor = TraceTensor(name, stack_info, int32, None, 1, None)

        if data is None:
            return

        if any(isinstance(elem, ShapeScalar) for elem in data):
            # Handle the case where data is a list of mixed int and ShapeScalar elements
            # Example: [1, a.shape[0]]
            shape = Shape([]) + data
            self.trace_tensor = shape.trace_tensor
        else:
            # TODO (pranavm): Figure out what to do for device?
            Storage.build_internal([], [self.trace_tensor], data, int32, device=None)

    # TODO (pranavm): Replace make Tensor constructor work with Shape directly - cannot use DLPack because it needs eval.
    #   Maybe copy underlying trace tensor.
    @constraints.dtypes(constraints={constraints.RETURN_VALUE: "T1"}, variables={"T1": ["int32"]})
    def as_tensor(self: "tripy.Shape") -> Tensor:
        """
        Return an ordinary Tripy :class:`Tensor` with the same contents as this :class:`Shape` . No copying is done.

        Args:
            self: This shape tensor.

        Returns:
            A :class:`Tensor` with the same underlying value as the current :class:`Shape` .

        .. code-block:: python
            :linenos:
            :caption: Example

            s = tp.Shape([1, 2, 3])
            t = s.as_tensor()
            assert isinstance(t, tp.Tensor) and not isinstance(t, tp.Shape)
            assert np.array_equal(cp.from_dlpack(s).get(), cp.from_dlpack(t).get())
        """
        ret = Tensor(data=None, dtype=int32, name=self.name, device=self.device)
        ret.trace_tensor = self.trace_tensor
        ret.stack_info = self.stack_info
        return ret

    # TODO (pranavm): Add `add` and `mul` as tp APIs that work for both shapes and tensors (using overloads).

    def _repr_name(self) -> str:
        return "shape"

    def __str__(self) -> str:
        return "shape" + "(" + ", ".join(map(str, self.tolist())) + ")"

    # addition for shapes is concatenation, not tensor addition

    def _validate_add_argument(self, other):
        if isinstance(other, Shape):
            return
        if not isinstance(other, Sequence) or (len(other) != 0 and not isinstance(other[0], int)):
            raise_error(
                "Invalid types for addition with a Tripy Shape.",
                details=[
                    "Implicit conversions are done only for sequences of Python ints. ",
                    "Consider calling tp.Shape for an explicit conversion. ",
                    f"Note: argument was {other}.",
                ],
            )

    def __add__(self, other: Union["tripy.Shape", Sequence[Union[int, ShapeScalar]]]):
        # TODO (pranavm): Make concatenate work for shapes
        from tripy.frontend.trace.ops.concatenate import concatenate

        # TODO: Make reshape work for ShapeScalar - it should return a Shape
        from tripy.frontend.trace.ops.reshape import reshape

        if isinstance(other, Shape):
            return concatenate([self, other], dim=0)

        # This should now be a list of shapes:
        elems = [reshape(elem, (1,)) if isinstance(elem, ShapeScalar) else Shape([elem]) for elem in other]
        return concatenate([self] + elems, dim=0)

    def __radd__(self, other: Union["tripy.Shape", Sequence[Union[int, ShapeScalar]]]):
        return Shape(other) + self

    def __mul__(self, other: int):
        # multiplication for shapes is tiling, not elementwise multiplication
        shape = Shape([])
        for _ in range(other):
            shape = shape + self
        return shape

    def __rmul__(self, other):
        return self * other

    @frontend_utils.convert_inputs_to_shapes(["other"])
    def __eq__(self, other):
        from tripy.frontend.trace.ops.reduce import all

        if len(self) != len(other):
            return False

        return bool(all(self.as_tensor() == other.as_tensor()))

    def __ne__(self, other):
        return not (self == other)

    # TODO (pranavm): This should come from the shape of the trace tensor.
    # __len__ for shapes gives the number of dims in the shape, i.e., the first dimension of the shape's shape
    def __len__(self):
        from tripy.frontend.trace.ops import utils as op_utils

        return op_utils.get_trace_shape(self.trace_tensor)[0]

    # TODO (pranavm): Check if this is still needed given that we define `__len__` and `__getitem__`.
    class ShapeIter:
        """
        Since it is generally simple to get the length of a shape, we can define
        iteration for Shapes even if we don't permit it for other tensors.
        """

        def __init__(self, shape):
            self.shape = shape
            self.idx = 0

        def __next__(self):
            if self.idx >= len(self.shape):
                raise StopIteration

            ret = self.shape[self.idx]
            self.idx += 1
            return ret

    def __iter__(self):
        return Shape.ShapeIter(self)
