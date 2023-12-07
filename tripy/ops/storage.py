from functools import reduce
from typing import List, Optional, Tuple

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
        data: List[int] or List[float],
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
        import tripy.common.datatype
        from tripy.common import device as make_device
        from tripy.frontend.dim import Dim

        self.device = util.default(device, make_device("cpu"))

        assert data is not None or dtype is not None
        assert data is not None or shape is not None

        # Flatten the list before storing it. If shape is not known, we just treat it is a 1-D list.
        e_type = tripy.common.datatype.float32
        if data is not None:
            data = util.flatten(data)
            assert all(isinstance(item, type(data[0])) for item in data)
            if len(data) > 0:
                t = type(data[0])
                assert t == float or t == int
                e_type = tripy.common.datatype.float32 if t == float else tripy.common.datatype.int32

        # For now require that required data type is same storage type.
        # Remove this restriction when cast operation is removed.
        assert dtype is None or e_type == dtype
        self.dtype = dtype or e_type

        shape = (len(data),) if shape is None else shape
        if data is None:
            assert shape is not None
            static_shape = [s.max if isinstance(s, Dim) else s for s in shape]
            nb_elements = reduce(lambda x, y: x * y, static_shape)
            data = [0] * nb_elements
        self.data = Array(data, self.dtype, self.device)
        self.shape = util.make_list(shape)
        self.shape_profile: List = util.make_list(shape)

    def __eq__(self, other) -> bool:
        return self.data == other.data

    def to_flat_ir_str(self, input_names, output_names):
        assert not input_names, "Storage should have no inputs!"
        assert len(output_names) == 1, "Storage should have exactly one output!"

        return f"{output_names[0]} : data=({self.data.view(self.dtype).tolist()}), shape=({self.shape}), dtype=({self.dtype.name}), stride=(), loc=({self.device.kind}:{self.device.index})"

    def infer_shapes(self, input_shapes):
        assert not input_shapes, "Storage should have no inputs!"
        return [util.make_tuple(self.shape)]

    def infer_dtypes(self, input_dtypes):
        assert not input_dtypes, "Storage should have no inputs!"
        return [self.dtype]

    def to_mlir(self, inputs):
        from tripy.backend.mlir import utils as mlir_utils

        assert not inputs, "Storage should have no inputs!"
        attr = ir.DenseElementsAttr.get(
            array=self.data.view(self.dtype), type=mlir_utils.get_mlir_dtype(self.dtype), shape=self.shape
        )
        return [stablehlo.ConstantOp(attr)]


@TENSOR_METHOD_REGISTRY("__init__")
def tensor_init(
    self: "tripy.Tensor",
    data: List[float] or List[int] = None,
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
