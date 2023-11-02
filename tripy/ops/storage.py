from typing import Any

import numpy as np
from mlir import ir
from mlir.dialects import stablehlo

from tripy.ops.base import BaseOperator


class Storage(BaseOperator):
    """
    Represents data stored in host memory.
    """

    # TODO (#10): We should have a custom storage class here instead of depending on NumPy.
    def __init__(self, data: Any):
        self.data = np.array(data)

    def __eq__(self, other) -> bool:
        return np.array_equal(self.data, other.data)

    def to_flat_ir_str(self, input_names, output_names):
        assert not input_names, "Storage should have no inputs!"
        assert len(output_names) == 1, "Storage should have exactly one output!"

        return f"{output_names[0]} : data=({self.data}), shape=(), stride=(), loc=()"

    def infer_shapes(self, input_shapes):
        assert not input_shapes, "Storage should have no inputs!"
        return [self.data.shape]

    def to_mlir(self, inputs):
        assert not inputs, "Storage should have no inputs!"
        # TODO (#11): Support non-FP32 types here.
        array = np.array(self.data, dtype=np.float32)
        attr = ir.DenseElementsAttr.get(np.ascontiguousarray(array), type=ir.F32Type.get(), shape=array.shape)
        return [stablehlo.ConstantOp(attr)]
