from typing import Any

import cupy as cp
import numpy as np
from mlir import ir
from mlir.dialects import stablehlo

from tripy import util
from tripy.ops.base import BaseOperator
from tripy.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.device import device


class Storage(BaseOperator):
    """
    Represents data stored in host memory.
    """

    # TODO (#10): We should have a custom storage class here instead of depending on NumPy.
    def __init__(self, data: Any, device: "tripy.frontend.Device" = device("cpu")):
        self._module = np if device.kind == "cpu" else cp
        # TODO (#11): Support non-FP32 types here.
        # TODO (#21): Support multiple devices
        self.data = self._module.array(data, dtype=self._module.float32)
        self.device = device

    def __eq__(self, other) -> bool:
        return self._module.array_equal(self.data, other.data)

    def to_flat_ir_str(self, input_names, output_names):
        assert not input_names, "Storage should have no inputs!"
        assert len(output_names) == 1, "Storage should have exactly one output!"

        return (
            f"{output_names[0]} : data=({self.data}), shape=(), stride=(), loc=({self.device.kind}:{self.device.index})"
        )

    def infer_shapes(self, input_shapes):
        assert not input_shapes, "Storage should have no inputs!"
        return [self.data.shape]

    def to_mlir(self, inputs):
        assert not inputs, "Storage should have no inputs!"
        # TODO (#11): Support non-FP32 types here.
        array = self._module.array(self.data, dtype=self._module.float32)
        attr = ir.DenseElementsAttr.get(self._module.ascontiguousarray(array), type=ir.F32Type.get(), shape=array.shape)
        return [stablehlo.ConstantOp(attr)]


@TENSOR_METHOD_REGISTRY("__init__")
def tensor_init(self: "tripy.Tensor", data: Any = None, device: "tripy.frontend.Device" = None) -> None:
    # Note: It is important that we are able to call the Tensor constructor with no arguments
    # since this is used internally by Tensor.build()
    if data is not None:
        from tripy.frontend import device as make_device
        from tripy.ops import Storage

        device = util.default(device, make_device("cpu"))
        self._finalize([], Storage(data, device))
