import enum
from dataclasses import dataclass

from tripy.common import datatype
from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.util import make_tuple, is_broadcast_compatible, get_broadcast_dim
from tripy.frontend.dim import Dim


@dataclass
class BinaryElementwise(BaseOperator):
    """
    Represents a binary elementwise operation.
    """

    class Kind(str, enum.Enum):
        SUM = " + "
        """Perform an elementwise sum"""
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

    def to_trace_str(self, input_names, output_names):
        assert len(output_names) == 1, "BinaryElementwise should have exactly one output!"
        return f"{output_names[0]} = {self.kind.join(input_names)}"

    def infer_shapes(self, input_shapes):
        # Fix when broadcasting support is added (#25).
        assert len(input_shapes[0]) == len(
            input_shapes[1]
        ), f"Input rank for BinaryElementwise operator do not match. Got {input_shapes[0]} and {input_shapes[1]}."

        assert is_broadcast_compatible(*input_shapes)

        return [make_tuple([get_broadcast_dim(*d) for d in zip(*input_shapes)])]

    def infer_dtypes(self, input_dtypes):
        assert (
            input_dtypes[0] == input_dtypes[1]
        ), f"Input data types for BinaryElementwise must match. Got: {input_dtypes[0]} and {input_dtypes[1]}"
        if self.kind in self._COMPARE_OPS:
            return [datatype.bool]
        return [input_dtypes[0]]

    def to_flat_ir(self, flat_ir, inputs, outputs):
        from tripy.flat_ir.ops import AddOp, CompareOp, BroadcastOp

        _MLIR_COMPARE_DIRECTIONS = {
            BinaryElementwise.Kind.LESS: "LT",
            BinaryElementwise.Kind.LESS_EQUAL: "LE",
            BinaryElementwise.Kind.EQUAL: "EQ",
            BinaryElementwise.Kind.NOT_EQUAL: "NE",
            BinaryElementwise.Kind.GREATER_EQUAL: "GE",
            BinaryElementwise.Kind.GREATER: "GT",
        }

        dynamic_shape = False
        requires_broadcast = False
        for dim1, dim2 in zip(inputs[0].shape, inputs[1].shape):
            requires_broadcast |= dim1 != dim2
            if dim1.is_dynamic_dim() or dim2.is_dynamic_dim():
                dynamic_shape = True

        def add_broadcast(self, flat_ir, inp, out):
            temp = flat_ir.add_tensor(out)
            flat_ir.ops.append(BroadcastOp(self, [inp], [temp], broadcast_dim=list(range(len(inp.shape)))))
            return temp

        if requires_broadcast:
            if not dynamic_shape:
                inputs[0] = add_broadcast(self, flat_ir, inputs[0], outputs[0])
                inputs[1] = add_broadcast(self, flat_ir, inputs[1], outputs[0])
            else:
                assert False, "Broadcast support with dynamic shapes is not enabled."

        if self.kind in self._COMPARE_OPS:
            flat_ir.ops.append(CompareOp(self, inputs, outputs, compare_direction=_MLIR_COMPARE_DIRECTIONS[self.kind]))
        elif self.kind == BinaryElementwise.Kind.SUM:
            flat_ir.ops.append(AddOp(self, inputs, outputs))
        else:
            raise NotImplementedError()


@TENSOR_METHOD_REGISTRY("__add__")
def add(self: "tripy.Tensor", other: "tripy.Tensor") -> "tripy.Tensor":
    """
    Performs an elementwise sum.

    Args:
        other: The tensor to add to this one.

    Returns:
        The sum of the inputs.

    Example:
    ::

        import numpy as np

        a = tp.Tensor([1, 2])
        b = tp.Tensor([2, 3])
        out = a + b
        assert (out.numpy() == np.array([3, 5])).all()
    """
    from tripy.frontend import Tensor

    return Tensor.build([self, other], BinaryElementwise, BinaryElementwise.Kind.SUM)


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

        import numpy as np

        a = tp.Tensor([2, 3])
        b = tp.Tensor([1, 5])
        out = b < a
        # TODO(#26): replace with out.numpy() after MLIR-TRT can handle i1 dtype's allocation
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

        import numpy as np

        a = tp.Tensor([2, 3])
        b = tp.Tensor([2, 5])
        out = b <= a
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

        import numpy as np

        a = tp.Tensor([2, 3])
        b = tp.Tensor([2, 5])
        out = b == a
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

        import numpy as np

        a = tp.Tensor([2, 3])
        b = tp.Tensor([1, 3])
        out = b != a
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

        import numpy as np

        a = tp.Tensor([2, 3])
        b = tp.Tensor([2, 1])
        out = b >= a
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

        import numpy as np

        a = tp.Tensor([2, 3])
        b = tp.Tensor([3, 1])
        out = b > a
        assert out.eval().view().tolist() == [True, False]
    """
    from tripy.frontend import Tensor

    return Tensor.build([self, other], BinaryElementwise, BinaryElementwise.Kind.GREATER)
