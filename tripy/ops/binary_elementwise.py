import enum
from dataclasses import dataclass
from typing import List

from mlir.dialects import stablehlo

from tripy.util import make_tuple
from tripy.ops.base import BaseOperator
from tripy.ops.registry import TENSOR_METHOD_REGISTRY


@dataclass
class BinaryElementwise(BaseOperator):
    """
    Represents a binary elementwise operation.
    """

    class Kind(enum.Enum):
        SUM = 0
        """Perform an elementwise sum"""

    kind: Kind
    """The operation to apply"""

    def to_flat_ir_str(self, input_names, output_names):
        assert self.kind == BinaryElementwise.Kind.SUM, "Only SUM is supported for now!"
        assert len(output_names) == 1, "BinaryElementwise should have exactly one output!"
        return f"{output_names[0]} = {' + '.join(input_names)}"

    def infer_shapes(self, input_shapes):
        assert self.kind == BinaryElementwise.Kind.SUM, "Only SUM is supported for now!"
        # Fix when broadcasting support is added (#25).
        assert (
            input_shapes[0] == input_shapes[1]
        ), f"Input shapes for BinaryElementwise operator do not match. Got {input_shapes[0]} and {input_shapes[1]}."
        return [make_tuple(input_shapes[0])]

    def to_mlir(self, inputs: List) -> List:
        assert self.kind == BinaryElementwise.Kind.SUM, "Only Operation.SUM is supported by MLIR backend."
        add_out = stablehlo.AddOp(*inputs)
        return [add_out]


@TENSOR_METHOD_REGISTRY("__add__")
def add(self: "tripy.Tensor", other: "tripy.Tensor") -> "tripy.Tensor":
    # TODO (#8): Add an example here.
    """
    Performs an elementwise sum.

    Args:
        other: The tensor to add to this one.

    Returns:
        The sum of the inputs.
    """
    from tripy.frontend import Tensor

    return Tensor.build(
        [self, other],
        BinaryElementwise(BinaryElementwise.Kind.SUM),
    )
