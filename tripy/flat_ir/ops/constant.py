import dataclasses
from dataclasses import dataclass
from typing import Any

from mlir import ir
from mlir.dialects import stablehlo

import tripy.common
from tripy.flat_ir.ops.base import BaseFIROp


@dataclass
class ConstantOp(BaseFIROp):
    """
    Operation to store a constant tensor
    """

    data: Any
    dtype: "tripy.dtype"
    device: tripy.common.device

    def __init__(self, origin_layer, inputs, outputs, data):
        super().__init__(origin_layer, inputs, outputs)
        assert len(self.outputs) == 1, "ConstantOp should have exactly 1 output"
        self.data = data
        self.dtype = self.outputs[0].dtype
        self.device = self.outputs[0].device

    def __str__(self) -> str:
        from tripy import utils

        data_volume = utils.volume(self.data.shape)

        skip_fields = [base_field.name for base_field in dataclasses.fields(BaseFIROp)]
        if utils.skip_constant_from_logging(data_volume):
            skip_fields = ["data"] + skip_fields

        args = [
            f"{field.name}={getattr(self, field.name)}"
            for field in dataclasses.fields(self)
            if field.name not in skip_fields
        ]
        return f"{self.outputs[0]} = {self.name()}({', '.join([inp.name for inp in self.inputs] + args)})"

    def to_mlir(self, operands):
        from tripy.backend.mlir import utils as mlir_utils

        attr = ir.DenseElementsAttr.get(
            array=self.data, type=mlir_utils.get_mlir_dtype(self.dtype), shape=self.data.shape
        )
        return [stablehlo.ConstantOp(attr)]
