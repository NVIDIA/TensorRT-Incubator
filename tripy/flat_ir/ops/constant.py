from dataclasses import dataclass
from typing import Set

from mlir_tensorrt.compiler import ir
from mlir_tensorrt.compiler.dialects import stablehlo

from tripy import utils
from tripy.common.array import Array
from tripy.flat_ir.ops.base import BaseFlatIROp

import mlir_tensorrt.runtime.api as runtime


@dataclass(repr=False)
class ConstantOp(BaseFlatIROp):

    data: Array

    def str_skip_fields(self) -> Set[str]:
        if utils.should_omit_constant_in_str(self.data.shape):
            return {"data"}
        return set()

    def to_mlir(self, operands):
        from tripy.backend.mlir import utils as mlir_utils

        assert isinstance(self.data, Array)
        memref_value = self.data.memref_value
        if self.data.device.kind == "gpu":
            memref_value = runtime.RuntimeClient().copy_to_host(
                device_memref=memref_value,
                stream=None,
            )
        attr = ir.DenseElementsAttr.get(
            array=memref_value, type=mlir_utils.get_mlir_dtype(self.outputs[0].dtype), shape=self.data.shape
        )

        return [stablehlo.ConstantOp(attr)]
