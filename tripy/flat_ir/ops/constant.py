from dataclasses import dataclass
from typing import Any, Set

from mlir import ir
from mlir.dialects import stablehlo

import tripy.common
from tripy import utils
from tripy.flat_ir.ops.base import BaseFlatIROp
from tripy.common.array import Array


@dataclass(repr=False)
class ConstantOp(BaseFlatIROp):

    data: Array
    dtype: "tripy.dtype"
    device: tripy.common.device

    def __init__(self, source_op, inputs, outputs, data):
        super().__init__(source_op, inputs, outputs)
        assert len(self.outputs) == 1, "ConstantOp should have exactly 1 output"
        assert isinstance(data, Array), "ConstantOp requires data to be initialized using Array class"
        self.data = data
        self.shape = utils.default(self.data.shape, (0,))
        self.dtype = self.outputs[0].dtype
        self.device = self.outputs[0].device

    def str_skip_fields(self) -> Set[str]:
        if utils.should_omit_constant_in_str(self.shape):
            return {"data"}
        return set()

    def to_mlir(self, operands):
        from tripy.backend.mlir import utils as mlir_utils
        import cupy as cp

        data = self.data.view()
        if isinstance(data, cp.ndarray):
            # This is required because MLIR-TRT backend requires constants to be on host.
            data = data.get()

        attr = ir.DenseElementsAttr.get(
            array=b"" if self.shape == (0,) else data, type=mlir_utils.get_mlir_dtype(self.dtype), shape=self.shape
        )
        return [stablehlo.ConstantOp(attr)]
