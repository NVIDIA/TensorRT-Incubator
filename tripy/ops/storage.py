from typing import Any, List

import cupy as cp
import numpy as np
from mlir import ir
from mlir.dialects import stablehlo

from tripy import util
from tripy.ops.base import BaseOperator
from tripy.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.named_dim import NamedDim


class Storage(BaseOperator):
    """
    Represents data stored in host memory.
    """

    # TODO (#10): We should have a custom storage class here instead of depending on NumPy.
    def __init__(self, data: Any, device: "tripy.frontend.Device" = None, shape: List = None):
        from tripy.frontend import device as make_device

        self.device = util.default(device, make_device("cpu"))
        self._module = np if self.device.kind == "cpu" else cp
        # TODO (#11): Support non-FP32 types here.
        # TODO (#21): Support multiple devices
        self.data = self._module.array(data, dtype=self._module.float32)
        shape = util.make_tuple(shape)
        self.shape: List = self.data.shape if shape is None else [-1 if isinstance(s, NamedDim) else s for s in shape]
        self.shape_profile: List = shape

    def __eq__(self, other) -> bool:
        return self._module.array_equal(self.data, other.data)

    def to_flat_ir_str(self, input_names, output_names):
        assert not input_names, "Storage should have no inputs!"
        assert len(output_names) == 1, "Storage should have exactly one output!"

        return f"{output_names[0]} : data=({self.data}), shape=({self.shape}), stride=(), loc=({self.device.kind}:{self.device.index})"

    def infer_shapes(self, input_shapes):
        assert not input_shapes, "Storage should have no inputs!"
        return [util.make_tuple(self.shape)]

    def to_mlir(self, inputs):
        assert not inputs, "Storage should have no inputs!"
        # TODO (#11): Support non-FP32 types here.
        attr = ir.DenseElementsAttr.get(
            self._module.ascontiguousarray(self.data), type=ir.F32Type.get(), shape=self.data.shape
        )
        return [stablehlo.ConstantOp(attr)]


@TENSOR_METHOD_REGISTRY("__init__")
def tensor_init(
    self: "tripy.Tensor", data: Any = None, device: "tripy.frontend.Device" = None, shape: List = None
) -> None:
    # Note: It is important that we are able to call the Tensor constructor with no arguments
    # since this is used internally by Tensor.build()
    if data is not None:
        from tripy.frontend import device as make_device
        from tripy.ops import Storage

        device = util.default(device, make_device("cpu"))
        self._finalize([], Storage(data, device, shape))
