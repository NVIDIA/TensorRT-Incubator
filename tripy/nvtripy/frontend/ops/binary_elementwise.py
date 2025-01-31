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


from nvtripy import export
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.frontend.ops._registry import register_tensor_method
from nvtripy.trace.ops.binary_elementwise import BinaryElementwise, Comparison
from nvtripy.types import TensorLike
from nvtripy.utils import wrappers


@register_tensor_method("__add__")
@register_tensor_method("__radd__")
@wrappers.interface(
    dtype_constraints={"self": "T1", "other": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64", "bool"]},
    aliases=["__radd__"],
    convert_to_tensors=True,
)
def __add__(self: "nvtripy.Tensor", other: TensorLike) -> "nvtripy.Tensor":
    """
    Performs an elementwise sum.

    Args:
        self: Tensor to be added to other.
        other: The tensor to add to this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:

        a = tp.Tensor([1, 2])
        b = tp.Tensor([2, 3])
        output = a + b

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([3, 5]))
    """
    return op_utils.create_op(BinaryElementwise, [self, other], BinaryElementwise.Kind.SUM)


@register_tensor_method("__sub__")
@wrappers.interface(
    dtype_constraints={"self": "T1", "other": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64"]},
    convert_to_tensors=True,
)
def __sub__(self: "nvtripy.Tensor", other: TensorLike) -> "nvtripy.Tensor":
    """
    Performs an elementwise subtraction.

    Args:
        self: Tensor to be subtracted by other.
        other: The tensor to subtract from this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:

        a = tp.Tensor([2, 3])
        b = tp.Tensor([1, 2])
        output = a - b

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([1, 1]))
    """
    return op_utils.create_op(BinaryElementwise, [self, other], BinaryElementwise.Kind.SUB)


@register_tensor_method("__rsub__")
@wrappers.interface(
    dtype_constraints={"self": "T1", "other": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64"]},
    convert_to_tensors=True,
)
def __rsub__(self: "nvtripy.Tensor", other: TensorLike) -> "nvtripy.Tensor":
    """
    Performs an elementwise subtraction.

    Args:
        self: Tensor to be subtracted by other.
        other: The tensor to be subtracted from this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:

        a = 1
        b = tp.Tensor([1, 2])
        output = a - b

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([0, -1]))
    """
    return op_utils.create_op(BinaryElementwise, [other, self], BinaryElementwise.Kind.SUB)


@register_tensor_method("__pow__")
@wrappers.interface(
    dtype_constraints={"self": "T1", "other": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8"]},
    convert_to_tensors=True,
)
def __pow__(self: "nvtripy.Tensor", other: TensorLike) -> "nvtripy.Tensor":
    """
    Performs an elementwise exponentiation.

    Args:
        self: Tensor to be exponentiated by other.
        other: The tensor by which to exponentiate this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:

        a = tp.Tensor([1.0, 2.0])
        b = tp.Tensor([2.0, 3.0])
        output = a ** b

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([1, 8]))
    """
    return op_utils.create_op(BinaryElementwise, [self, other], BinaryElementwise.Kind.POW)


@register_tensor_method("__rpow__")
@wrappers.interface(
    dtype_constraints={"self": "T1", "other": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8"]},
    convert_to_tensors=True,
)
def __rpow__(self: "nvtripy.Tensor", other: TensorLike) -> "nvtripy.Tensor":
    """
    Performs an elementwise exponentiation.

    Args:
        self: Tensor to be exponentiated by other.
        other: The tensor to be exponentiated by this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:

        a = 2.0
        b = tp.Tensor([2.0, 3.0])
        output = a ** b

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([4.0, 8.0]))
    """
    return op_utils.create_op(BinaryElementwise, [other, self], BinaryElementwise.Kind.POW)


@register_tensor_method("__mul__")
@register_tensor_method("__rmul__")
@wrappers.interface(
    dtype_constraints={"self": "T1", "other": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64", "bool"]},
    aliases=["__rmul__"],
    convert_to_tensors=True,
)
def __mul__(self: "nvtripy.Tensor", other: TensorLike) -> "nvtripy.Tensor":
    """
    Performs an elementwise multiplication.

    Args:
        self: Tensor to be multiplied by other.
        other: The tensor by which to multiply this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:

        a = tp.Tensor([1.0, 2.0])
        b = tp.Tensor([2.0, 3.0])
        output = a * b

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([2.0, 6.0]))
    """
    return op_utils.create_op(BinaryElementwise, [self, other], BinaryElementwise.Kind.MUL)


@register_tensor_method("__truediv__")
@wrappers.interface(
    dtype_constraints={"self": "T1", "other": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64"]},
    convert_to_tensors=True,
)
def __truediv__(self: "nvtripy.Tensor", other: TensorLike) -> "nvtripy.Tensor":
    """
    Performs an elementwise division.

    Args:
        self: Tensor to be divided by other.
        other: The tensor by which to divide this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:

        a = tp.Tensor([4.0, 6.0])
        b = tp.Tensor([2.0, 3.0])
        output = a / b

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([2.0, 2.0]))
    """
    return op_utils.create_op(BinaryElementwise, [self, other], BinaryElementwise.Kind.DIV)


@register_tensor_method("__rtruediv__")
@wrappers.interface(
    dtype_constraints={"self": "T1", "other": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64"]},
    convert_to_tensors=True,
)
def __rtruediv__(self: "nvtripy.Tensor", other: TensorLike) -> "nvtripy.Tensor":
    """
    Performs an elementwise division.

    Args:
        self: Tensor to be divided by other.
        other: The tensor to be divided by this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:

        a = 6.0
        b = tp.Tensor([2.0, 3.0])
        output = a / b

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([3.0, 2.0]))
    """
    return op_utils.create_op(BinaryElementwise, [other, self], BinaryElementwise.Kind.DIV)


@register_tensor_method("__floordiv__")
@wrappers.interface(
    dtype_constraints={"self": "T1", "other": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64"]},
    convert_to_tensors=True,
)
def __floordiv__(self: "nvtripy.Tensor", other: TensorLike) -> "nvtripy.Tensor":
    """
    Performs an elementwise floor division.

    Args:
        self: Tensor to be floor-divided by other.
        other: The tensor by which to floor-divide this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:

        a = tp.Tensor([4.0, 6.0])
        b = tp.Tensor([3.0, 4.0])
        output = a // b

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([1.0, 1.0]))
    """
    from nvtripy.common.datatype import int32
    from nvtripy.frontend.ops.cast import cast

    return cast(
        cast(op_utils.create_op(BinaryElementwise, [self, other], BinaryElementwise.Kind.DIV), int32), self.dtype
    )
    # Use the below code when https://github.com/NVIDIA/TensorRT-Incubator/issues/208 is fixed
    # return op_utils.create_op(BinaryElementwise, [self, other], BinaryElementwise.Kind.FLOOR_DIV)


@register_tensor_method("__rfloordiv__")
@wrappers.interface(
    dtype_constraints={"self": "T1", "other": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64"]},
    convert_to_tensors=True,
)
def __rfloordiv__(self: "nvtripy.Tensor", other: TensorLike) -> "nvtripy.Tensor":
    """
    Performs an elementwise floor division.

    Args:
        self: Tensor to be floor-divided by other.
        other: The tensor to be floor-divided by this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:

        a = 2
        b = tp.Tensor([2.0, 3.0])
        output = a // b

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([1.0, 0.0]))
    """
    from nvtripy.common.datatype import int32
    from nvtripy.frontend.ops.cast import cast

    return cast(
        cast(op_utils.create_op(BinaryElementwise, [other, self], BinaryElementwise.Kind.DIV), int32), self.dtype
    )
    # Use the below code when https://github.com/NVIDIA/TensorRT-Incubator/issues/208 is fixed
    # return op_utils.create_op(BinaryElementwise, [other, self], BinaryElementwise.Kind.FLOOR_DIV)


@register_tensor_method("__mod__")
@wrappers.interface(
    dtype_constraints={"self": "T1", "other": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16"]},
    convert_to_tensors=True,
)
def __mod__(self: "nvtripy.Tensor", other: TensorLike) -> "nvtripy.Tensor":
    """
    Performs a modulo operation.

    Args:
        self: The tensor to be divided by `other`.
        other: The tensor by which to divide `self`.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape containing the result of the modulo operation.

    .. code-block:: python
        :linenos:

        a = tp.Tensor([4.0, 6.0])
        b = tp.Tensor([3.0, 4.0])
        output = a % b

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([1.0, 2.0]))
    """
    return op_utils.create_op(BinaryElementwise, [self, other], BinaryElementwise.Kind.MOD)


@register_tensor_method("__rmod__")
@wrappers.interface(
    dtype_constraints={"self": "T1", "other": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16"]},
    convert_to_tensors=True,
)
def __rmod__(self: "nvtripy.Tensor", other: TensorLike) -> "nvtripy.Tensor":
    """
    Performs a modulo operation.

    Args:
        self: The tensor to be divided by `other`.
        other: The tensor by which to divide `self`.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape containing the result of the modulo operation.

    .. code-block:: python
        :linenos:

        a = tp.Tensor([4.0, 6.0])
        output = 2 % a

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([2.0, 2.0]))
    """
    return op_utils.create_op(BinaryElementwise, [other, self], BinaryElementwise.Kind.MOD)


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"lhs": "T1", "rhs": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"]},
)
def maximum(lhs: "nvtripy.Tensor", rhs: "nvtripy.Tensor") -> "nvtripy.Tensor":
    """
    Performs an elementwise maximum.

    Args:
        lhs: The first input tensor.
        rhs: The second input tensor.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:

        a = tp.Tensor([1.0, 6.0])
        b = tp.Tensor([2.0, 3.0])
        output = tp.maximum(a, b)

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([2.0, 6.0]))
    """
    return op_utils.create_op(BinaryElementwise, [lhs, rhs], BinaryElementwise.Kind.MAXIMUM)


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"lhs": "T1", "rhs": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"]},
)
def minimum(lhs: "nvtripy.Tensor", rhs: "nvtripy.Tensor") -> "nvtripy.Tensor":
    """
    Performs an elementwise minimum.

    Args:
        lhs: The first input tensor.
        rhs: The second input tensor.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:

        a = tp.Tensor([1.0, 6.0])
        b = tp.Tensor([2.0, 3.0])
        output = tp.minimum(a, b)

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([1.0, 3.0]))
    """
    return op_utils.create_op(BinaryElementwise, [lhs, rhs], BinaryElementwise.Kind.MINIMUM)


@register_tensor_method("__lt__")
@wrappers.interface(
    dtype_constraints={"self": "T1", "other": "T1", wrappers.RETURN_VALUE: "T2"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
        "T2": ["bool"],
    },
    convert_to_tensors=True,
)
def __lt__(self: "nvtripy.Tensor", other: TensorLike) -> "nvtripy.Tensor":
    """
    Performs a 'less than' comparison.

    Args:
        self: Tensor to be compared with other.
        other: The tensor to be compared to this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:

        a = tp.Tensor([2, 3])
        b = tp.Tensor([1, 5])
        output = b < a

        assert output.tolist() == [True, False]
    """
    return op_utils.create_op(Comparison, [self, other], Comparison.Kind.LESS)


@register_tensor_method("__le__")
@wrappers.interface(
    dtype_constraints={"self": "T1", "other": "T1", wrappers.RETURN_VALUE: "T2"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
        "T2": ["bool"],
    },
    convert_to_tensors=True,
)
def __le__(self: "nvtripy.Tensor", other: TensorLike) -> "nvtripy.Tensor":
    """
    Performs a 'less than or equal' comparison.

    Args:
        self: Tensor to be compared with other.
        other: The tensor to be compared to this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:

        a = tp.Tensor([2, 3])
        b = tp.Tensor([2, 5])
        output = b <= a

        assert output.tolist() == [True, False]
    """
    return op_utils.create_op(Comparison, [self, other], Comparison.Kind.LESS_EQUAL)


@register_tensor_method("__eq__")
@wrappers.interface(
    dtype_constraints={"self": "T1", "other": "T1", wrappers.RETURN_VALUE: "T2"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
        "T2": ["bool"],
    },
    convert_to_tensors=True,
)
def __eq__(self: "nvtripy.Tensor", other: TensorLike) -> "nvtripy.Tensor":
    """
    Performs an 'equal' comparison.

    Args:
        self: Tensor to be compared with other.
        other: The tensor to be compared to this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:

        a = tp.Tensor([2, 3])
        b = tp.Tensor([2, 5])
        output = b == a

        assert output.tolist() == [True, False]
    """
    return op_utils.create_op(Comparison, [self, other], Comparison.Kind.EQUAL)


@register_tensor_method("__ne__")
@wrappers.interface(
    dtype_constraints={"self": "T1", "other": "T1", wrappers.RETURN_VALUE: "T2"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
        "T2": ["bool"],
    },
    convert_to_tensors=True,
)
def __ne__(self: "nvtripy.Tensor", other: TensorLike) -> "nvtripy.Tensor":
    """
    Performs a 'not equal' comparison.

    Args:
        self: Tensor to be compared with other.
        other: The tensor to be compared to this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:

        a = tp.Tensor([2, 3])
        b = tp.Tensor([1, 3])
        output = b != a

        assert output.tolist() == [True, False]
    """
    return op_utils.create_op(Comparison, [self, other], Comparison.Kind.NOT_EQUAL)


@register_tensor_method("__ge__")
@wrappers.interface(
    dtype_constraints={"self": "T1", "other": "T1", wrappers.RETURN_VALUE: "T2"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
        "T2": ["bool"],
    },
    convert_to_tensors=True,
)
def __ge__(self: "nvtripy.Tensor", other: TensorLike) -> "nvtripy.Tensor":
    """
    Performs a 'greater than or equal' comparison.

    Args:
        self: Tensor to be compared with other.
        other: The tensor to be compared to this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:

        a = tp.Tensor([2, 3])
        b = tp.Tensor([2, 1])
        output = b >= a

        assert output.tolist() == [True, False]
    """
    return op_utils.create_op(Comparison, [self, other], Comparison.Kind.GREATER_EQUAL)


@register_tensor_method("__gt__")
@wrappers.interface(
    dtype_constraints={"self": "T1", "other": "T1", wrappers.RETURN_VALUE: "T2"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
        "T2": ["bool"],
    },
    convert_to_tensors=True,
)
def __gt__(self: "nvtripy.Tensor", other: TensorLike) -> "nvtripy.Tensor":
    """
    Performs a 'greater than' comparison.

    Args:
        self: Tensor to be compared with other.
        other: The tensor to be compared to this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:

        a = tp.Tensor([2, 3])
        b = tp.Tensor([3, 1])
        output = b > a

        assert output.tolist() == [True, False]
    """
    return op_utils.create_op(Comparison, [self, other], Comparison.Kind.GREATER)
