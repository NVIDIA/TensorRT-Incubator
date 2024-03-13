from dataclasses import dataclass

from mlir_tensorrt.compiler.dialects import tensorrt

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class QuantizeOp(BaseFlatIROp):

    def to_mlir(self, operands):
        # out_quant_type = get_mlir_quant_dtype(
        #     self.inputs[0].dtype,
        #     self.outputs[0].dtype,
        #     self.scale,
        #     self.zero_point,
        #     self.storage_min,
        #     self.storage_max,
        # )
        # out_type = ir.RankedTensorType.get(
        #     [ir.ShapedType.get_dynamic_size() if s.is_dynamic_dim() else s.min for s in self.outputs[0].shape],
        #     out_quant_type,
        # )
        # return [stablehlo.uniform_quantize(out_type, operands[0])]
        out_type = self.outputs[0].to_mlir()
        return [tensorrt.quantize(out_type, *operands)]
