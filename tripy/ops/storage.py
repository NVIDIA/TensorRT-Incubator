from typing import List, Optional, Tuple, Union

import numpy as np
import cupy as cp
import jax.numpy as jnp
import torch

from mlir import ir
from mlir.dialects import stablehlo

from tripy import util
from tripy.common.array import Array
from tripy.ops.base import BaseOperator
from tripy.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.common.datatype import DataTypeConverter


class Storage(BaseOperator):
    """
    Represents data stored in host or device memory.
    """

    def __init__(
        self,
        data: Union[list, np.ndarray, cp.ndarray, torch.Tensor, jnp.ndarray],
        shape: Optional[Tuple[int]] = None,
        dtype: "tripy.common.DataType" = None,
        device: "tripy.common.Device" = None,
    ) -> None:
        """
        Initialize Storage instance.

        Args:
            data: The data to be stored.
            dtype: Data type (default: float32).
            device: The device where the data is stored (default: CPU).
            shape: The shape of the data (default: None).
        """
        from tripy.common import device as make_device
        from tripy.frontend.dim import Dim

        self.device = util.default(device, make_device("cpu"))
        self.data = Array(data, dtype, shape, self.device)
        self.dtype = self.data.dtype
        self.shape: Tuple[int] = util.make_tuple(self.data.shape if shape is None else shape)
        self.shape_profile: List = util.make_list(shape)

    def __eq__(self, other) -> bool:
        return self.data == other.data

    def to_flat_ir_str(self, input_names, output_names):
        assert not input_names, "Storage should have no inputs!"
        assert len(output_names) == 1, "Storage should have exactly one output!"

        return f"{output_names[0]} : data=({self.data.view(self.dtype).tolist()}), shape=({self.shape}), dtype=({self.dtype.name}), stride=(), loc=({self.device.kind}:{self.device.index})"

    def infer_shapes(self, input_shapes):
        assert not input_shapes, "Storage should have no inputs!"
        return [util.make_tuple(self.data.shape)]

    def infer_dtypes(self, input_dtypes):
        assert not input_dtypes, "Storage should have no inputs!"
        return [self.dtype]

    def to_mlir(self, inputs):
        from tripy.backend.mlir import utils as mlir_utils

        assert not inputs, "Storage should have no inputs!"
        attr = ir.DenseElementsAttr.get(
            array=self.data.view(self.dtype), type=mlir_utils.get_mlir_dtype(self.dtype), shape=self.data.shape
        )
        return [stablehlo.ConstantOp(attr)]


@TENSOR_METHOD_REGISTRY("__init__")
def tensor_init(
    self: "tripy.Tensor",
    data: Union[list, np.ndarray, cp.ndarray, torch.Tensor, jnp.ndarray] = None,
    shape: Optional[Tuple[int]] = None,
    dtype: "tripy.common.DataType" = None,
    device: "tripy.common.Device" = None,
) -> None:
    """
    Creates a tensor.

    Args:
        data: The data with which to initialize the tensor.
        shape: The shape of the tensor.
        dtype: The data type of the tensor.
        device: The device on which to allocate the tensor.

    Example:
    ::

        import tripy

        tensor = tripy.Tensor([1.0, 2.0, 3.0], shape=(3,) , dtype=tripy.float32)
    """
    # Note: It is important that we are able to call the Tensor constructor with no arguments
    # since this is used internally by Tensor.build()
    if data is not None:
        from tripy.ops import Storage

        self._finalize([], Storage(data, shape, dtype, device))
