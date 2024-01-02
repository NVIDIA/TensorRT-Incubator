from mlir import ir
from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFIROp


class ConstantOp(BaseFIROp):
    """
    Operation to store a constant
    """

    def __init__(self, origin_layer, inputs, outputs, **kwargs):
        super().__init__(inputs, outputs, origin_layer)
        assert "data" in kwargs
        self.data = kwargs.get("data")

    def to_flat_ir_str(self, input_names, output_names) -> str:
        data = self.origin_layer.data.view()
        return f"{output_names[0]} : {self.__class__.__name__} data=({data.view()}), shape=({data.shape}), dtype=({self.origin_layer.dtype.name}), loc=({self.origin_layer.device.kind}:{self.origin_layer.device.index})"

    def to_mlir(self, operands):
        from tripy.backend.mlir import utils as mlir_utils

        attr = ir.DenseElementsAttr.get(
            array=self.data, type=mlir_utils.get_mlir_dtype(self.origin_layer.dtype), shape=self.data.shape
        )
        return [stablehlo.ConstantOp(attr)]
