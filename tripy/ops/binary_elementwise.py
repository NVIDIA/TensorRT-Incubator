import enum
from dataclasses import dataclass

from tripy.ops.base import BaseOperator


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
        return [input_shapes[0]]
