from mlir import ir
from mlir.dialects import arith, stablehlo

from tripy.flat_ir.ops.base import BaseFIROp


class ConstantOp(BaseFIROp):
    """
    Operation to store a constant tensor
    """

    def __init__(self, origin_layer, inputs, outputs, **kwargs):
        super().__init__(inputs, outputs, origin_layer)
        assert "data" in kwargs
        assert len(self.outputs) == 1, "ConstantOp should have exactly 1 output"
        self.data = kwargs.get("data")
        self.dtype = self.outputs[0].dtype
        self.device = self.outputs[0].device

    def to_flat_ir_str(self, input_names, output_names) -> str:
        return f"{output_names[0]} : {self.__class__.__name__} data={self.data}, shape={self.data.shape}, dtype={self.dtype.name}, loc=({self.device.kind}:{self.device.index})"

    def to_mlir(self, operands):
        from tripy.backend.mlir import utils as mlir_utils

        attr = ir.DenseElementsAttr.get(
            array=self.data, type=mlir_utils.get_mlir_dtype(self.dtype), shape=self.data.shape
        )
        return [stablehlo.ConstantOp(attr)]


class ConstantScalarOp(BaseFIROp):
    """
    Operation to store a constant scalar
    """

    def __init__(self, origin_layer, inputs, outputs, **kwargs):
        super().__init__(inputs, outputs, origin_layer)
        assert "value" in kwargs and "dtype" in kwargs
        self.value = kwargs.get("value")
        self.dtype = kwargs.get("dtype")

    def to_flat_ir_str(self, input_names, output_names) -> str:
        return f"{output_names[0]} : {self.__class__.__name__} data={self.value}, shape={self.data.shape}, dtype={self.dtype.name}, loc={self.device}"

    def to_mlir(self, operands):
        from tripy.backend.mlir import utils as mlir_utils

        mlir_dtype = mlir_utils.get_mlir_dtype(self.dtype)
        if isinstance(mlir_dtype, ir.IntegerType):
            const_attr = ir.IntegerAttr.get(
                type=mlir_dtype,
                value=self.value,
            )
        else:
            const_attr = ir.FloatAttr.get(
                type=mlir_dtype,
                value=self.value,
            )
        return [arith.constant(const_attr)]
