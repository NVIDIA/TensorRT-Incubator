from dataclasses import dataclass

from mlir_tensorrt.compiler.dialects import tensorrt

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class QuantizeOp(BaseFlatIROp):

    def to_mlir(self, operands):
        # TODO(#111): switch to stablehlo after mlir-trt supports it
        out_type = self.outputs[0].to_mlir()
        return [tensorrt.quantize(out_type, *operands)]
