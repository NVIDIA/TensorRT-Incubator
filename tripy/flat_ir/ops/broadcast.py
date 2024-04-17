from dataclasses import dataclass
from typing import List

from mlir_tensorrt.compiler import ir
from mlir_tensorrt.compiler.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp
import array


@dataclass(repr=False)
class BroadcastOp(BaseFlatIROp):
    broadcast_dim: List[int]

    def to_mlir(self, operands):
        out_type = self.outputs[0].to_mlir()
        broadcast_dim_attr = ir.DenseI64ArrayAttr.get(self.broadcast_dim)
        output = stablehlo.broadcast_in_dim(out_type, operands[0], broadcast_dim_attr)
        return [output]


@dataclass(repr=False)
class DynamicBroadcastOp(BaseFlatIROp):

    broadcast_dim: List[int]

    def to_mlir(self, operands):

        broadcast_dim_attr = ir.DenseI64ArrayAttr.get(self.broadcast_dim)
        out_type = self.outputs[0].to_mlir()

        output = stablehlo.dynamic_broadcast_in_dim(out_type, operands[0], operands[1], broadcast_dim_attr)
        return [output]
