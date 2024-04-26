from dataclasses import dataclass
from typing import Set

from mlir_tensorrt.compiler import ir
from mlir_tensorrt.compiler.dialects import stablehlo

from tripy import utils
from tripy.common.array import Array, convert_tripy_to_module_dtype
from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class ConstantOp(BaseFlatIROp):

    data: Array

    def str_skip_fields(self) -> Set[str]:
        if utils.should_omit_constant_in_str(self.data.shape):
            return {"data"}
        return set()

    def to_mlir(self, operands):
        import cupy as cp
        import numpy as np

        from tripy.backend.mlir import utils as mlir_utils

        assert isinstance(self.data, Array)
        # HACK(#169): Remove cupy .get()
        data = self.data.byte_buffer
        if isinstance(data, cp.ndarray):
            # This is required because MLIR-TRT backend requires constants to be on host.
            data = data.get()
        data = data.view(convert_tripy_to_module_dtype(self.data.dtype, np))

        attr = ir.DenseElementsAttr.get(
            array=data, type=mlir_utils.get_mlir_dtype(self.outputs[0].dtype), shape=self.data.shape
        )

        return [stablehlo.ConstantOp(attr)]
