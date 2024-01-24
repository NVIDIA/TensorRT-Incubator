import enum
from dataclasses import dataclass

import tripy.frontend.ops.utils as op_utils
from tripy.common import datatype
from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY


@dataclass
class BinaryElementwise(BaseOperator):
    """
    Represents a binary elementwise operation.
    """

    class Kind(str, enum.Enum):
        SUM = " + "
        """Perform an elementwise sum"""
        SUB = " - "
        """Perform an elementwise subtraction"""
        POW = " ** "
        """Perform an elementwise power"""
        MUL = " * "
        """Perform an elementwise multiplication"""
        DIV = " / "
        """Perform an elementwise divide"""
        LESS = " < "
        """Perform a 'less than' comparison"""
        LESS_EQUAL = " <= "
        """Perform a 'less than or equal' comparison"""
        EQUAL = " == "
        """Perform an 'equal' comparison"""
        NOT_EQUAL = " != "
        """Perform a 'not equal' comparison"""
        GREATER_EQUAL = " >= "
        """Perform a 'greater than or equal' comparison"""
        GREATER = " > "
        """Perform a 'greater than' comparison"""

    kind: Kind
    """The operation to apply"""

    _COMPARE_OPS = {
        Kind.LESS,
        Kind.LESS_EQUAL,
        Kind.EQUAL,
        Kind.NOT_EQUAL,
        Kind.GREATER_EQUAL,
        Kind.GREATER,
    }

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
                    self.kind.value.strip(),
                    "' must be broadcast compatible but ",
                ]
                + bcast_check.details,
            )
        self.outputs[0].shape = tuple(op_utils.get_broadcast_dim(*d) for d in zip(*input_shapes))

    def infer_dtypes(self):
        op_utils.check_input_dtypes_match(self, self.kind.value.strip())
        self.outputs[0].dtype = datatype.bool if self.kind in self._COMPARE_OPS else self.inputs[0].dtype

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import AddOp, CompareOp, DivideOp, MulOp, PowOp, SubtractOp

        _MLIR_COMPARE_DIRECTIONS = {
            BinaryElementwise.Kind.LESS: "LT",
            BinaryElementwise.Kind.LESS_EQUAL: "LE",
            BinaryElementwise.Kind.EQUAL: "EQ",
            BinaryElementwise.Kind.NOT_EQUAL: "NE",
            BinaryElementwise.Kind.GREATER_EQUAL: "GE",
            BinaryElementwise.Kind.GREATER: "GT",
        }

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

        if self.kind in self._COMPARE_OPS:
            CompareOp(self, inputs, outputs, compare_direction=_MLIR_COMPARE_DIRECTIONS[self.kind])
        else:
            OpType = {
                BinaryElementwise.Kind.SUM: AddOp,
                BinaryElementwise.Kind.POW: PowOp,
                BinaryElementwise.Kind.MUL: MulOp,
                BinaryElementwise.Kind.SUB: SubtractOp,
                BinaryElementwise.Kind.DIV: DivideOp,
            }[self.kind]
            OpType(self, inputs, outputs)


@TENSOR_METHOD_REGISTRY("__add__")
@op_utils.allow_non_tensor
def add(self: "tripy.Tensor", other: "tripy.Tensor") -> "tripy.Tensor":
    """
    Performs an elementwise sum.

    Args:
        other: The tensor to add to this one.

    Returns:
        The sum of the inputs.

    Example:
    ::

        a = tp.Tensor([1, 2])
        b = tp.Tensor([2, 3])
        out = a + b
        print(out)
        assert np.array_equal(out.numpy(), np.array([3, 5]))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self, other], BinaryElementwise, BinaryElementwise.Kind.SUM)


@TENSOR_METHOD_REGISTRY("__sub__")
@op_utils.allow_non_tensor
def sub(self: "tripy.Tensor", other: "tripy.Tensor") -> "tripy.Tensor":
    """
    Performs an elementwise subtraction.

    Args:
        other: The tensor to subtract from this one.

    Returns:
        Result of subtraction

    Example:
    ::

        a = tp.Tensor([2, 3])
        b = tp.Tensor([1, 2])
        out = a - b
        print(out)
        assert np.array_equal(out.numpy(), np.array([1, 1]))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self, other], BinaryElementwise, BinaryElementwise.Kind.SUB)


@TENSOR_METHOD_REGISTRY("__pow__")
@op_utils.allow_non_tensor
def pow(self: "tripy.Tensor", other: "tripy.Tensor") -> "tripy.Tensor":
    """
    Performs an elementwise pow.

    Args:
        other: The tensor by which to exponentiate this one.

    Returns:
        Result of exponentiation.

    Example:
    ::

        a = tp.Tensor([1.0, 2.0])
        b = tp.Tensor([2.0, 3.0])
        out = a ** b
        print(out)
        assert np.array_equal(out.numpy(), np.array([1, 8]))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self, other], BinaryElementwise, BinaryElementwise.Kind.POW)


# #84 will add __rmul__: @TENSOR_METHOD_REGISTRY("__rmul__")
@TENSOR_METHOD_REGISTRY("__mul__")
@op_utils.allow_non_tensor
def mul(self: "tripy.Tensor", other: "tripy.Tensor") -> "tripy.Tensor":
    """
    Performs an elementwise multiplication.

    Args:
        other: The tensor by which to multiply this one.

    Returns:
        Product of two tensors

    Example:
    ::

        a = tp.Tensor([1.0, 2.0])
        b = tp.Tensor([2.0, 3.0])
        out = a * b
        print(out)
        assert np.array_equal(out.numpy(), np.array([2.0, 6.0]))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self, other], BinaryElementwise, BinaryElementwise.Kind.MUL)


@TENSOR_METHOD_REGISTRY("__truediv__")
def div(self: "tripy.Tensor", other: "tripy.Tensor") -> "tripy.Tensor":
    """
    Performs an elementwise divide.

    Args:
        other: The tensor by which to divide this one.

    Returns:
        Result of divide

    Example:
    ::

        a = tp.Tensor([4.0, 6.0])
        b = tp.Tensor([2.0, 3.0])
        out = a / b
        print(out)
        assert np.array_equal(out.numpy(), np.array([2.0, 2.0]))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self, other], BinaryElementwise, BinaryElementwise.Kind.DIV)


@TENSOR_METHOD_REGISTRY("__lt__")
def less_than(self: "tripy.Tensor", other: "tripy.Tensor") -> "tripy.Tensor":
    """
    Performs a 'less than' comparison.

    Args:
        other: The tensor to be compared to this one

    Returns:
        The comparison result of the inputs

    Example:
    ::

        a = tp.Tensor([2, 3])
        b = tp.Tensor([1, 5])
        out = b < a
        # TODO(#26): replace with out.numpy() after MLIR-TRT can handle i1 dtype's allocation
        print(out)
        assert out.eval().view().tolist() == [True, False]
    """
    from tripy.frontend import Tensor

    return Tensor.build([self, other], BinaryElementwise, BinaryElementwise.Kind.LESS)


@TENSOR_METHOD_REGISTRY("__le__")
def less_than_or_equal(self: "tripy.Tensor", other: "tripy.Tensor") -> "tripy.Tensor":
    """
    Performs a 'less than or equal' comparison.

    Args:
        other: The tensor to be compared to this one

    Returns:
        The comparison result of the inputs

    Example:
    ::

        a = tp.Tensor([2, 3])
        b = tp.Tensor([2, 5])
        out = b <= a
        print(out)
        assert out.eval().view().tolist() == [True, False]
    """
    from tripy.frontend import Tensor

    return Tensor.build([self, other], BinaryElementwise, BinaryElementwise.Kind.LESS_EQUAL)


@TENSOR_METHOD_REGISTRY("__eq__")
def eq(self: "tripy.Tensor", other: "tripy.Tensor") -> "tripy.Tensor":
    """
    Performs an 'equal' comparison.

    Args:
        other: The tensor to be compared to this one

    Returns:
        The comparison result of the inputs

    Example:
    ::

        a = tp.Tensor([2, 3])
        b = tp.Tensor([2, 5])
        out = b == a
        print(out)
        assert out.eval().view().tolist() == [True, False]
    """
    from tripy.frontend import Tensor

    return Tensor.build([self, other], BinaryElementwise, BinaryElementwise.Kind.EQUAL)


@TENSOR_METHOD_REGISTRY("__ne__")
def not_equal(self: "tripy.Tensor", other: "tripy.Tensor") -> "tripy.Tensor":
    """
    Performs a 'not equal' comparison.

    Args:
        other: The tensor to be compared to this one

    Returns:
        The comparison result of the inputs

    Example:
    ::

        a = tp.Tensor([2, 3])
        b = tp.Tensor([1, 3])
        out = b != a
        print(out)
        assert out.eval().view().tolist() == [True, False]
    """
    from tripy.frontend import Tensor

    return Tensor.build([self, other], BinaryElementwise, BinaryElementwise.Kind.NOT_EQUAL)


@TENSOR_METHOD_REGISTRY("__ge__")
def greater_than_or_equal(self: "tripy.Tensor", other: "tripy.Tensor") -> "tripy.Tensor":
    """
    Performs a 'greater than or equal' comparison.

    Args:
        other: The tensor to be compared to this one

    Returns:
        The comparison result of the inputs

    Example:
    ::

        a = tp.Tensor([2, 3])
        b = tp.Tensor([2, 1])
        out = b >= a
        print(out)
        assert out.eval().view().tolist() == [True, False]
    """
    from tripy.frontend import Tensor

    return Tensor.build([self, other], BinaryElementwise, BinaryElementwise.Kind.GREATER_EQUAL)


@TENSOR_METHOD_REGISTRY("__gt__")
def greater_than(self: "tripy.Tensor", other: "tripy.Tensor") -> "tripy.Tensor":
    """
    Performs a 'greater than' comparison.

    Args:
        other: The tensor to be compared to this one

    Returns:
        The comparison result of the inputs

    Example:
    ::

        a = tp.Tensor([2, 3])
        b = tp.Tensor([3, 1])
        out = b > a
        print(out)
        assert out.eval().view().tolist() == [True, False]
    """
    from tripy.frontend import Tensor

    return Tensor.build([self, other], BinaryElementwise, BinaryElementwise.Kind.GREATER)
