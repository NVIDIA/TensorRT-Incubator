import enum
from dataclasses import dataclass
from typing import Any, Union

import tripy.frontend.ops.utils as op_utils
from tripy.common import datatype
from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
import tripy.frontend.utils as frontend_utils


@dataclass(repr=False)
class BinaryElementwise(BaseOperator):
    """
    Represents a binary elementwise operation.
    """

    class Kind:
        SUM = " + "
        SUB = " - "
        POW = " ** "
        MUL = " * "
        DIV = " / "

    kind: str

    def __str__(self):
        return f"{self.outputs[0].name} = {self.kind.join([inp.name for inp in self.inputs])}"

    def infer_shapes(self):
        input_shapes = [inp.shape for inp in self.inputs]
        input_shapes = op_utils.get_broadcast_compatible_shapes(self.inputs[0].shape, self.inputs[1].shape)
        bcast_check = op_utils.is_broadcast_compatible(*input_shapes)
        if not bcast_check:
            op_utils.raise_error_io_info(
                self,
                "Input tensors are not broadcast compatible.",
                details=[
                    "Input tensors for binary operation: '",
                    self.kind.strip(),
                    "' must be broadcast compatible but ",
                ]
                + bcast_check.details,
            )
        self.outputs[0].shape = tuple(op_utils.get_broadcast_dim(*d) for d in zip(*input_shapes))

    def infer_dtypes(self):
        op_utils.check_input_dtypes_match(self, self.kind.strip())
        self.outputs[0].dtype = self.inputs[0].dtype

    def broadcast_inputs(self, inputs, outputs):
        # Insert broadcast to ensure operands are of the same rank.
        shape1, shape2 = op_utils.get_broadcast_compatible_shapes(inputs[0].shape, inputs[1].shape)

        dynamic_shape = False
        requires_broadcast_1 = shape1 != inputs[0].shape
        requires_broadcast_2 = shape2 != inputs[1].shape

        for dim1, dim2 in zip(shape1, shape2):
            if dim1 != dim2:
                requires_broadcast_1 |= dim1 < dim2
                requires_broadcast_2 |= dim1 > dim2
            if dim1.is_dynamic_dim() or dim2.is_dynamic_dim():
                dynamic_shape = True

        if dynamic_shape and (requires_broadcast_1 or requires_broadcast_2):
            assert False, "Broadcast support with dynamic shapes is not enabled."

        if requires_broadcast_1:
            inputs[0] = op_utils.insert_broadcast(self, inputs[0], outputs[0].shape)
        if requires_broadcast_2:
            inputs[1] = op_utils.insert_broadcast(self, inputs[1], outputs[0].shape)

        return inputs

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import AddOp, DivideOp, MulOp, PowOp, SubtractOp

        inputs = self.broadcast_inputs(inputs, outputs)
        OpType = {
            BinaryElementwise.Kind.SUM: AddOp,
            BinaryElementwise.Kind.POW: PowOp,
            BinaryElementwise.Kind.MUL: MulOp,
            BinaryElementwise.Kind.SUB: SubtractOp,
            BinaryElementwise.Kind.DIV: DivideOp,
        }[self.kind]
        OpType(self, inputs, outputs)


@dataclass(repr=False)
class Comparison(BinaryElementwise):
    class Kind:
        class KindElem(str):
            def __new__(cls, content, compare_direction):
                instance = super().__new__(cls, content)
                instance.compare_direction = compare_direction
                return instance

        LESS = KindElem(" < ", "LT")
        LESS_EQUAL = KindElem(" <= ", "LE")
        EQUAL = KindElem(" == ", "EQ")
        NOT_EQUAL = KindElem(" != ", "NE")
        GREATER_EQUAL = KindElem(" >= ", "GE")
        GREATER = KindElem(" > ", "GT")

    kind: Kind.KindElem

    def infer_dtypes(self):
        op_utils.check_input_dtypes_match(self, self.kind.strip())
        self.outputs[0].dtype = datatype.bool

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import CompareOp

        inputs = self.broadcast_inputs(inputs, outputs)
        CompareOp(self, inputs, outputs, compare_direction=self.kind.compare_direction)


@TENSOR_METHOD_REGISTRY("__add__")
@TENSOR_METHOD_REGISTRY("__radd__")
@frontend_utils.convert_inputs_to_tensors()
def add(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise sum.

    Args:
        other: The tensor to add to this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and of the same data type as the inputs.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([1, 2])
        b = tp.Tensor([2, 3])
        output = a + b

        assert np.array_equal(output.numpy(), np.array([3, 5]))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self, other], BinaryElementwise, BinaryElementwise.Kind.SUM)


@TENSOR_METHOD_REGISTRY("__sub__")
@frontend_utils.convert_inputs_to_tensors()
def sub(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise subtraction.

    Args:
        other: The tensor to subtract from this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and of the same data type as the inputs.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([2, 3])
        b = tp.Tensor([1, 2])
        output = a - b

        assert np.array_equal(output.numpy(), np.array([1, 1]))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self, other], BinaryElementwise, BinaryElementwise.Kind.SUB)


@TENSOR_METHOD_REGISTRY("__rsub__")
@frontend_utils.convert_inputs_to_tensors()
def rsub(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise subtraction.

    Args:
        other: The tensor to be subtracted from this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and of the same data type as the inputs.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = 1
        b = tp.Tensor([1, 2])
        output = a - b

        assert np.array_equal(output.numpy(), np.array([0, -1]))
    """
    from tripy.frontend import Tensor

    return Tensor.build([other, self], BinaryElementwise, BinaryElementwise.Kind.SUB)


@TENSOR_METHOD_REGISTRY("__pow__")
@frontend_utils.convert_inputs_to_tensors()
def pow(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise exponentiation.

    Args:
        other: The tensor by which to exponentiate this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and of the same data type as the inputs.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([1.0, 2.0])
        b = tp.Tensor([2.0, 3.0])
        output = a ** b

        assert np.array_equal(output.numpy(), np.array([1, 8]))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self, other], BinaryElementwise, BinaryElementwise.Kind.POW)


@TENSOR_METHOD_REGISTRY("__rpow__")
@frontend_utils.convert_inputs_to_tensors()
def pow(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise exponentiation.

    Args:
        other: The tensor to be exponentiated by this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and of the same data type as the inputs.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = 2.0
        b = tp.Tensor([2.0, 3.0])
        output = a ** b

        assert np.array_equal(output.numpy(), np.array([4.0, 8.0]))
    """
    from tripy.frontend import Tensor

    return Tensor.build([other, self], BinaryElementwise, BinaryElementwise.Kind.POW)


@TENSOR_METHOD_REGISTRY("__mul__")
@TENSOR_METHOD_REGISTRY("__rmul__")
@frontend_utils.convert_inputs_to_tensors()
def mul(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise multiplication.

    Args:
        other: The tensor by which to multiply this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and of the same data type as the inputs.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([1.0, 2.0])
        b = tp.Tensor([2.0, 3.0])
        output = a * b

        assert np.array_equal(output.numpy(), np.array([2.0, 6.0]))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self, other], BinaryElementwise, BinaryElementwise.Kind.MUL)


@TENSOR_METHOD_REGISTRY("__truediv__")
@frontend_utils.convert_inputs_to_tensors()
def div(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise division.

    Args:
        other: The tensor by which to divide this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and of the same data type as the inputs.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([4.0, 6.0])
        b = tp.Tensor([2.0, 3.0])
        output = a / b

        assert np.array_equal(output.numpy(), np.array([2.0, 2.0]))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self, other], BinaryElementwise, BinaryElementwise.Kind.DIV)


@TENSOR_METHOD_REGISTRY("__rtruediv__")
@frontend_utils.convert_inputs_to_tensors()
def div(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise division.

    Args:
        other: The tensor to be divided by this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and of the same data type as the inputs.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = 6.0
        b = tp.Tensor([2.0, 3.0])
        output = a / b

        assert np.array_equal(output.numpy(), np.array([3.0, 2.0]))
    """
    from tripy.frontend import Tensor

    return Tensor.build([other, self], BinaryElementwise, BinaryElementwise.Kind.DIV)


@TENSOR_METHOD_REGISTRY("__lt__")
@frontend_utils.convert_inputs_to_tensors()
def less_than(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs a 'less than' comparison.

    Args:
        other: The tensor to be compared to this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and datatype :class:`tripy.bool`.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([2, 3])
        b = tp.Tensor([1, 5])
        output = b < a
        # TODO(#26): replace with output.numpy() after MLIR-TRT can handle i1 dtype's allocation

        assert output.eval().view().tolist() == [True, False]
    """
    from tripy.frontend import Tensor

    return Tensor.build([self, other], Comparison, Comparison.Kind.LESS)


@TENSOR_METHOD_REGISTRY("__le__")
@frontend_utils.convert_inputs_to_tensors()
def less_than_or_equal(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs a 'less than or equal' comparison.

    Args:
        other: The tensor to be compared to this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and datatype :class:`tripy.bool`.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([2, 3])
        b = tp.Tensor([2, 5])
        output = b <= a

        assert output.eval().view().tolist() == [True, False]
    """
    from tripy.frontend import Tensor

    return Tensor.build([self, other], Comparison, Comparison.Kind.LESS_EQUAL)


@TENSOR_METHOD_REGISTRY("__eq__")
@frontend_utils.convert_inputs_to_tensors()
def eq(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an 'equal' comparison.

    Args:
        other: The tensor to be compared to this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and datatype :class:`tripy.bool`.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([2, 3])
        b = tp.Tensor([2, 5])
        output = b == a

        assert output.eval().view().tolist() == [True, False]
    """
    from tripy.frontend import Tensor

    return Tensor.build([self, other], Comparison, Comparison.Kind.EQUAL)


@TENSOR_METHOD_REGISTRY("__ne__")
@frontend_utils.convert_inputs_to_tensors()
def not_equal(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs a 'not equal' comparison.

    Args:
        other: The tensor to be compared to this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and datatype :class:`tripy.bool`.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([2, 3])
        b = tp.Tensor([1, 3])
        output = b != a

        assert output.eval().view().tolist() == [True, False]
    """
    from tripy.frontend import Tensor

    return Tensor.build([self, other], Comparison, Comparison.Kind.NOT_EQUAL)


@TENSOR_METHOD_REGISTRY("__ge__")
@frontend_utils.convert_inputs_to_tensors()
def greater_than_or_equal(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs a 'greater than or equal' comparison.

    Args:
        other: The tensor to be compared to this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and datatype :class:`tripy.bool`.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([2, 3])
        b = tp.Tensor([2, 1])
        output = b >= a

        assert output.eval().view().tolist() == [True, False]
    """
    from tripy.frontend import Tensor

    return Tensor.build([self, other], Comparison, Comparison.Kind.GREATER_EQUAL)


@TENSOR_METHOD_REGISTRY("__gt__")
@frontend_utils.convert_inputs_to_tensors()
def greater_than(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs a 'greater than' comparison.

    Args:
        other: The tensor to be compared to this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and datatype :class:`tripy.bool`.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([2, 3])
        b = tp.Tensor([3, 1])
        output = b > a

        assert output.eval().view().tolist() == [True, False]
    """
    from tripy.frontend import Tensor

    return Tensor.build([self, other], Comparison, Comparison.Kind.GREATER)
