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

from typing import Optional, Sequence, Union

from tripy import export, utils
from tripy.common.datatype import int32
from tripy.common.exception import raise_error
from tripy.frontend.tensor import Tensor
import tripy.frontend.utils as frontend_utils


@export.public_api()
class ShapeScalar(Tensor):
    """
    Scalar shape is a tensor used to represent a scalar value extracted from a shape tensor.
    ShapeScalars are scalars (rank 0) of non-negative integer (using int32 as the datatype).
    """

    def __init__(
        self,
        data: Union[Sequence, Tensor, "np.ndarray", "cp.ndarray", "torch.Tensor", "jnp.ndarray"],
        name: Optional[str] = None,
    ) -> None:
        r"""
        Args:
            data: The value of the ShapeScalar, which should be a scalar integer.
            name: An optional name
        """

        from tripy.common.exception import raise_error

        if isinstance(data, Tensor):
            # these fields can be None in the case of an uninitialized tensor (like Tensor(None))
            if data.trace_tensor.rank is not None and data.trace_tensor.rank != 0:
                raise_error(
                    f"Scalar shape tensors must be of rank 0, but input tensor is rank {data.rank}", details=[data]
                )
            if data.dtype is not None and data.dtype != int32:
                raise_error(
                    f"Scalar shape tensor must have int32 member, but input tensor has data type {data.dtype}",
                    details=[data],
                )

            # the shape of data should correspond to the given rank
            super().__init__(data=None, dtype=int32, name=name, device=data.device)
            # share the underlying data
            self.trace_tensor = data.trace_tensor
            self.stack_info = data.stack_info
        else:
            shape = data.shape if hasattr(data, "shape") else utils.get_shape(data)
            device = data.device if hasattr(data, "device") else None
            if len(shape) != 0:
                raise_error(
                    f"Tensors used to represent scalar shapes must be of rank 0, but given shape {shape} has rank {len(shape)}."
                )
            super().__init__(data=data, dtype=int32, name=name, device=device)

    def __repr__(self) -> str:
        # denote the representation as a shape rather than a tensor
        tensor_repr = super().__repr__()
        assert tensor_repr[:6] == "tensor"
        return "shape_scalar" + tensor_repr[6:]

    def __str__(self) -> str:
        val = self.tolist()
        assert isinstance(val, int)
        return f"shape_scalar({val})"


@export.public_api()
class Shape(Tensor):
    """
    A Shape is a tensor used to represent a tensor shape.
    Shapes are vectors (rank 1) of non-negative integers (using int32 as the datatype).

    Note that Shapes are intended to be used in many cases like Python lists; hence `+` acts as concatenation
    on Shapes rather than elementwise addition and `*` acts as tiling rather than elementwise multiplication;
    the methods `add` and `multiply` can be used for elementwise addition and multiplication, respectively.
    Additionally, `len` can be used to get the length of a shape.
    """

    # set to higher than tensor's so adding a shape to a tensor will use shape's overload (and give an error)
    __array_priority__ = 20000

    def __init__(
        self,
        data: Union[Sequence, Tensor, "np.ndarray", "cp.ndarray", "torch.Tensor", "jnp.ndarray"],
        name: Optional[str] = None,
    ) -> None:
        r"""
        Args:
            data: The value of the shape, which should be a 1D array of integers (the dimensions).
            name: An optional name
        """

        from tripy.common.exception import raise_error

        if isinstance(data, Tensor):
            # these fields can be None in the case of an uninitialized tensor (like Tensor(None))
            if data.trace_tensor.rank is not None and data.trace_tensor.rank != 1:
                raise_error(f"Shape tensors must be of rank 1, but input tensor is rank {data.rank}", details=[data])
            if data.dtype is not None and data.dtype != int32:
                raise_error(
                    f"Shape tensors must have int32 members, but input tensor has data type {data.dtype}",
                    details=[data],
                )

            # the shape of data should correspond to the given rank
            super().__init__(data=None, dtype=int32, name=name, device=data.device)
            # share the underlying data
            self.trace_tensor = data.trace_tensor
            self.stack_info = data.stack_info
        elif isinstance(data, Sequence) and len(data) > 0 and any(map(lambda e: isinstance(e, ShapeScalar), data)):
            # Handle the case where data is a list of mixed int and ShapeScalar elements
            # Example: [1, a.shape[0]]
            # We convert this to a tensor to avoid expensive evaluation of ShapeScalar elements (like a.shape[0])
            from tripy.frontend.trace.ops.concatenate import concatenate
            from tripy.frontend.trace.ops.reshape import reshape

            data = concatenate(
                [reshape(e, (1,)) if isinstance(e, ShapeScalar) else Tensor([e], dtype=int32) for e in data], dim=0
            )
            # the shape of data should correspond to the given rank
            super().__init__(data=None, dtype=int32, name=name, device=data.device)
            # share the underlying data
            self.trace_tensor = data.trace_tensor
            self.stack_info = data.stack_info
        else:
            shape = data.shape if hasattr(data, "shape") else utils.get_shape(data)
            device = data.device if hasattr(data, "device") else None
            if len(shape) != 1:
                raise_error(
                    f"Tensors used to represent shapes must be of rank 1, but given shape {shape} has rank {len(shape)}."
                )
            super().__init__(data=data, dtype=int32, name=name, device=device)

    def as_tensor(self) -> Tensor:
        """
        Return an ordinary Tripy :class:`Tensor` with the same contents as this :class:`Shape` . No copying is done.

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

    def add(self, other: Union["tripy.Shape", Tensor]) -> "tripy.Shape":
        """
        The + operator for shapes is concatenation. This method is exposed to allow for elementwise addition,
        should it be necessary.

        Args:
            other: Another :class:`Shape` or :class:`Tensor` .

        Returns:
            The result of elementwise addition of this :class:`Shape` and `other`, returned as a :class:`Shape` .

        .. code-block:: python
            :linenos:
            :caption: Example

            s1 = tp.Shape([1, 2, 3])
            s2 = tp.Shape([4, 5, 6])
            res = s1.add(s2)
            assert isinstance(res, tp.Shape)
            assert cp.from_dlpack(res).get().tolist() == [5, 7, 9]
        """
        return super().__add__(other)

    def multiply(self, other: Union["tripy.Shape", Tensor]) -> "tripy.Shape":
        """
        The * operator for shapes is tiling. This method is exposed to allow for elementwise multiplication,
        should it be necessary.

        Args:
            other: Another :class:`Shape` or :class:`Tensor` .

        Returns:
            The result of elementwise multiplication of this :class:`Shape` and `other`, returned as a :class:`Shape` .

        .. code-block:: python
            :linenos:
            :caption: Example

            s1 = tp.Shape([1, 2, 3])
            s2 = tp.Shape([4, 5, 6])
            res = s1.multiply(s2)
            assert isinstance(res, tp.Shape)
            assert cp.from_dlpack(res).get().tolist() == [4, 10, 18]
        """
        return super().__mul__(other)

    def __repr__(self) -> str:
        # denote the representation as a shape rather than a tensor
        tensor_repr = super().__repr__()
        assert tensor_repr[:6] == "tensor"
        return "shape" + tensor_repr[6:]

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

    def __add__(self, other):
        from tripy.frontend.trace.ops.concatenate import concatenate

        self._validate_add_argument(other)
        if not isinstance(other, Shape):
            other = Shape(other)
        return concatenate([self, other], 0)

    def __radd__(self, other):
        from tripy.frontend.trace.ops.concatenate import concatenate

        self._validate_add_argument(other)
        if not isinstance(other, Shape):
            other = Shape(other)
        return concatenate([other, self], 0)

    # multiplication for shapes is tiling, not elementwise multiplication

    def __mul__(self, other):
        from tripy.frontend.trace.ops.binary_elementwise import maximum
        from tripy.frontend.trace.ops.expand import expand
        from tripy.frontend.trace.ops.reshape import reshape, flatten
        from tripy.frontend.trace.ops.unsqueeze import unsqueeze

        # We unsqueeze self into shape [1, len(self)], so giving [other, len(self)] as
        # the argument to expand will result in a shape of [other, len(self)] by
        # copying self the correct number of times.

        # Only defined with a scalar argument
        if not isinstance(other, Tensor):
            # note: Python does not accept floats as arguments for list multiplication either
            if isinstance(other, Sequence):
                raise_error(
                    "Attempting to multiply a Tripy Shape by a sequence, which is undefined",
                    details=[f"Note: argument was {other}."],
                )
            if not isinstance(other, int):
                raise_error(
                    "Invalid types for multplication with a Tripy Shape.",
                    details=[
                        "Implicit conversions are done only for Python ints. ",
                        "Consider calling tp.Shape for an explicit conversion. ",
                        f"Note: argument was: {other}.",
                    ],
                )
            other = Tensor(other, dtype=int32)
        if other.rank >= 1:
            raise_error(
                "Attempting to multiply a Tripy Shape by a tensor of rank >= 1, which is undefined",
                details=[f"Note: argument was {other}."],
            )
        # note: in Python, if a list is multiplied by a negative number, this is the same as multiplying by 0,
        # so we should clamp the argument
        if other.rank == 0:
            other = reshape(other, (1,))
        other = Shape(maximum(other, Tensor(0, int32))) + [len(self)]

        unsqueezed = unsqueeze(self, 0)
        tiled = expand(unsqueezed, other)
        # flatten the result so we get back to a rank-1 shape
        ret = flatten(tiled)
        return Shape(ret)

    def __rmul__(self, other):
        return self.__mul__(other)

    @frontend_utils.convert_shape_inputs(["other"])
    def __eq__(self, other):
        from tripy.frontend.trace.ops.reduce import all

        if len(self) != len(other):
            return False

        return bool(all(self.as_tensor() == other.as_tensor()))

    def __ne__(self, other):
        return not (self == other)

    # __len__ for shapes gives the number of dims in the shape, i.e., the first dimension of the shape's shape
    def __len__(self):
        from tripy.frontend.trace.ops import utils as op_utils

        return op_utils.get_trace_shape(self.trace_tensor)[0]

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
