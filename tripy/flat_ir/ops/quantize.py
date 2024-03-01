from dataclasses import dataclass

from mlir import ir
from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp
from tripy.backend.mlir.utils import get_mlir_quant_dtype


@dataclass(repr=False)
class QuantizeOp(BaseFlatIROp):

    scale: float
    zero_point: int
    storage_min: int
    storage_max: int

    def to_mlir(self, operands):
        out_quant_type = get_mlir_quant_dtype(
            self.inputs[0].dtype,
            self.outputs[0].dtype,
            self.scale,
            self.zero_point,
            self.storage_min,
            self.storage_max,
        )
        out_type = ir.RankedTensorType.get(
            [ir.ShapedType.get_dynamic_size() if s.is_dynamic_dim() else s.min for s in self.outputs[0].shape],
            out_quant_type,
        )
        return [stablehlo.uniform_quantize(out_type, operands[0])]
