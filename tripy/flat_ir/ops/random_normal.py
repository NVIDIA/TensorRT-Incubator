import numbers
from dataclasses import dataclass

from mlir import ir

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class RandomNormalOp(BaseFlatIROp):
    """
    Operation to generate a random tensor with normal distribution.
    """

    static_mean: numbers.Number
    static_std: numbers.Number

    def __init__(self, origin_layer, inputs, outputs, static_mean, static_std):
        super().__init__(origin_layer, inputs, outputs)
        self.static_mean = static_mean
        self.static_std = static_std

    def to_mlir(self, operands):
        out_type = self.outputs[0].to_mlir()
        static_mean_attr = ir.FloatAttr.get_f64(float(self.static_mean))
        static_std_attr = ir.FloatAttr.get_f64(float(self.static_std))
        out = ir.Operation.create(
            f"tensorrt.random_normal",
            results=[out_type],
            operands=[],
            attributes={"static_mean": static_mean_attr, "static_std": static_std_attr},
        ).result
        return [out]
