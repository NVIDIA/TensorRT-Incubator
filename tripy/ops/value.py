from dataclasses import dataclass
from typing import Any

from tripy.ops.base import BaseOperator


@dataclass
class Value(BaseOperator):
    """
    Represents values stored in a tensor.
    """

    values: Any  # TODO: This should be a GPU-backed tensor.

    def shape(self):
        import numpy as np

        assert isinstance(self.values, list) or isinstance(self.values, np.ndarray)
        if isinstance(self.values, list):
            return [len(self.values)]
        elif isinstance(self.values, np.ndarray):
            return self.values.shape
        return None

    def to_flat_ir_str(self, input_names, output_names):
        assert not input_names, "ValueParameters should have no inputs!"
        assert len(output_names) == 1, "ValueParameters should have exactly one output!"

        return f"{output_names[0]} : values=({self.values}), shape=(), stride=(), loc=()"

    def infer_shapes(self, input_shapes):
        assert not input_shapes, "ValueParameters should have no inputs!"
        return [self.shape()]
