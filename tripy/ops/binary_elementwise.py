import enum
from dataclasses import dataclass
from typing import List

from mlir.dialects import stablehlo

from tripy.common import datatype
from tripy.util import make_tuple
from tripy.ops.base import BaseOperator
from tripy.ops.registry import TENSOR_METHOD_REGISTRY


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

    def to_flat_ir_str(self, input_names, output_names):
        assert len(output_names) == 1, "BinaryElementwise should have exactly one output!"
        return f"{output_names[0]} = {self.kind.join(input_names)}"

    def infer_shapes(self, input_shapes):
        # Fix when broadcasting support is added (#25).
        assert (
            input_shapes[0] == input_shapes[1]
        ), f"Input shapes for BinaryElementwise operator do not match. Got {input_shapes[0]} and {input_shapes[1]}."
        return [make_tuple(input_shapes[0])]

    def infer_dtypes(self, input_dtypes):
        assert (
            input_dtypes[0] == input_dtypes[1]
        ), f"Input data types for BinaryElementwise must match. Got: {input_dtypes[0]} and {input_dtypes[1]}"
        if self.kind in self._COMPARE_OPS:
            return [datatype.bool]
        return [input_dtypes[0]]

    def to_mlir(self, inputs: List) -> List:
        _MLIR_COMPARE_DIRECTIONS = {
            BinaryElementwise.Kind.LESS: stablehlo.ComparisonDirectionAttr.get("LT"),
            BinaryElementwise.Kind.LESS_EQUAL: stablehlo.ComparisonDirectionAttr.get("LE"),
            BinaryElementwise.Kind.EQUAL: stablehlo.ComparisonDirectionAttr.get("EQ"),
            BinaryElementwise.Kind.NOT_EQUAL: stablehlo.ComparisonDirectionAttr.get("NE"),
            BinaryElementwise.Kind.GREATER_EQUAL: stablehlo.ComparisonDirectionAttr.get("GE"),
            BinaryElementwise.Kind.GREATER: stablehlo.ComparisonDirectionAttr.get("GT"),
        }
        if self.kind in self._COMPARE_OPS:
            out = stablehlo.CompareOp(*inputs, _MLIR_COMPARE_DIRECTIONS[self.kind])
        elif self.kind == BinaryElementwise.Kind.SUM:
            out = stablehlo.AddOp(*inputs)
        else:
            raise NotImplementedError()
        return [out]


@TENSOR_METHOD_REGISTRY("__add__")
def add(self: "tripy.Tensor", other: "tripy.Tensor") -> "tripy.Tensor":
    # TODO (#8): Add an example here.
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

    return Tensor.build(
        [self, other],
        BinaryElementwise(BinaryElementwise.Kind.SUM),
    )


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
        assert (out.numpy() == np.array([True, False])).all()
    """
    from tripy.frontend import Tensor

    return Tensor.build(
        [self, other],
        BinaryElementwise(BinaryElementwise.Kind.LESS),
    )


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
        assert (out.numpy() == np.array([True, False])).all()
    """
    from tripy.frontend import Tensor

    return Tensor.build(
        [self, other],
        BinaryElementwise(BinaryElementwise.Kind.LESS_EQUAL),
    )


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
        assert (out.numpy() == np.array([True, False])).all()
    """
    from tripy.frontend import Tensor

    return Tensor.build(
        [self, other],
        BinaryElementwise(BinaryElementwise.Kind.EQUAL),
    )


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
        assert (out.numpy() == np.array([True, False])).all()
    """
    from tripy.frontend import Tensor

    return Tensor.build(
        [self, other],
        BinaryElementwise(BinaryElementwise.Kind.NOT_EQUAL),
    )


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
        assert (out.numpy() == np.array([True, False])).all()
    """
    from tripy.frontend import Tensor

    return Tensor.build(
        [self, other],
        BinaryElementwise(BinaryElementwise.Kind.GREATER_EQUAL),
    )


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
        assert (out.numpy() == np.array([True, False])).all()
    """
    from tripy.frontend import Tensor

    return Tensor.build(
        [self, other],
        BinaryElementwise(BinaryElementwise.Kind.GREATER),
    )
