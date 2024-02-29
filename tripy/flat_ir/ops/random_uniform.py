import numbers
from dataclasses import dataclass

from mlir import ir

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(init=False, repr=False)
class RandomUniformOp(BaseFlatIROp):

    static_low: numbers.Number
    static_high: numbers.Number

    def __init__(self, source_op, inputs, outputs, static_low, static_high):
        super().__init__(source_op, inputs, outputs)
        self.static_low = static_low
        self.static_high = static_high

    def to_mlir(self, operands):
        out_type = self.outputs[0].to_mlir()
        static_low_attr = ir.FloatAttr.get_f64(float(self.static_low))
        static_high_attr = ir.FloatAttr.get_f64(float(self.static_high))
        out = ir.Operation.create(
            f"tensorrt.random_uniform",
            results=[out_type],
            operands=[],
            attributes={"static_low": static_low_attr, "static_high": static_high_attr},
        ).result
        return [out]
