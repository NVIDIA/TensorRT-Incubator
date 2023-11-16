from typing import Any, List

import cupy as cp
import numpy as np
from mlir import ir
from mlir.dialects import stablehlo

from tripy import util
from tripy.ops.base import BaseOperator
from tripy.ops.registry import TENSOR_METHOD_REGISTRY

from tripy.backend.mlir.types import TensorShape


class Storage(BaseOperator):
    """
    Represents data stored in host or device memory.
    """

    def __init__(
        self,
        data: Any,
        dtype: "tripy.common.DataType" = None,
        device: "tripy.frontend.Device" = None,
        shape: List = None,
    ) -> None:
        """
        Initialize Storage instance.

        Args:
            data: The data to be stored.
            dtype: Data type (default: float32).
            device: The device where the data is stored (default: CPU).
            shape: The shape of the data (default: None).
        """
        import tripy.common.datatype
        from tripy.common import device as make_device
        from tripy.frontend.dim import Dim

        self.device = util.default(device, make_device("cpu"))
        self.dtype = util.default(dtype, tripy.common.datatype.float32)

        self._module = np if self.device.kind == "cpu" else cp

        # TODO (#21): Support multiple devices
        # TODO (#10): getattr mostly works here because our data type naming mostly matches numpy/cupy,
        #   but we will need to eventually update this for our custom storage implementation.
        def convert_dtype():
            if self.dtype == tripy.common.datatype.bool:
                return self._module.bool_
            return getattr(self._module, self.dtype.name)

        if self.device.kind == "gpu" and (
            isinstance(data, cp.cuda.MemoryPointer) or isinstance(data, cp.cuda.memory.PooledMemory)
        ):
            # Store device memory pointer directly. Element type and shape are encoded in self.dtype and self.shape.
            self.data = data
        else:
            self.data = self._module.array(data, dtype=convert_dtype())

        shape = util.make_tuple(shape)
        self.shape: List = self.data.shape if shape is None else [-1 if isinstance(s, Dim) else s for s in shape]
        self.shape_profile: List = shape

    def __eq__(self, other) -> bool:
        return self._module.array_equal(self.data, other.data)

    def to_flat_ir_str(self, input_names, output_names):
        assert not input_names, "Storage should have no inputs!"
        assert len(output_names) == 1, "Storage should have exactly one output!"

        return f"{output_names[0]} : data=({self.data}), shape=({self.shape}), dtype=({self.dtype.name}), stride=(), loc=({self.device.kind}:{self.device.index})"

    def infer_shapes(self, input_shapes):
        assert not input_shapes, "Storage should have no inputs!"
        return [util.make_tuple(self.shape)]

    def infer_dtypes(self, input_dtypes):
        assert not input_dtypes, "Storage should have no inputs!"
        return [self.dtype]

    def to_mlir(self, inputs):
        from tripy.backend.mlir import utils as mlir_utils

        assert not inputs, "Storage should have no inputs!"
        data = (
            np.ascontiguousarray(self._module.asnumpy(self.data))
            if self.device.kind == "gpu"
            else self._module.ascontiguousarray(self.data)
        )
        attr = ir.DenseElementsAttr.get(data, type=mlir_utils.convert_dtype(self.dtype), shape=self.data.shape)
        return [stablehlo.ConstantOp(attr)]

    def to_ctypes(self):
        shape = TensorShape(self.dtype(), self.shape)
        return Storage(self.data.data.mem, self.dtype, self.device, shape)


@TENSOR_METHOD_REGISTRY("__init__")
def tensor_init(
    self: "tripy.Tensor",
    data: Any = None,
    dtype: "tripy.common.DataType" = None,
    device: "tripy.frontend.Device" = None,
    shape: List = None,
) -> None:
    """
    Creates a tensor.

    Args:
        data: The data with which to initialize the tensor.
        dtype: The data type of the tensor.
        device: The device on which to allocate the tensor.
        shape: The shape of the tensor.

    Example:
    ::

        import tripy

        tensor = tripy.Tensor([1, 2, 3], dtype=tripy.float32)
    """
    # Note: It is important that we are able to call the Tensor constructor with no arguments
    # since this is used internally by Tensor.build()
    if data is not None:
        from tripy.ops import Storage

        self._finalize([], Storage(data, dtype, device, shape))
