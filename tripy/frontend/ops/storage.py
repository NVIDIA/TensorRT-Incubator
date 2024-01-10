from typing import List, Optional, Tuple, Union

from tripy import util
from tripy.common.types import ShapeInfo
from tripy.common import device as make_device
from tripy.common.array import Array
from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.utils import to_dims
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY


class Storage(BaseOperator):
    """
    Represents data stored in host or device memory.
    """

    def __init__(
        self,
        data: Union[List, "np.ndarray", "cp.ndarray", "torch.Tensor", "jnp.ndarray"],
        shape: Optional[Tuple["Dim"]] = None,
        dtype: "tripy.dtype" = None,
        device: "tripy.common.device" = None,
    ) -> None:
        """
        Initialize Storage instance.

        Args:
            data: The data to be stored.
            dtype: Data type (default: float32).
            device: The device where the data is stored (default: CPU).
            shape: The shape of the data (default: None).
        """
        # Let's not allow user to request a different type unless data is a list.
        if hasattr(data, "to_dlpack"):
            # Ensure that dtype is not set.
            assert dtype is None

        self.device = util.default(device, make_device("cpu"))
        self.data = Array(data, dtype, shape, self.device)
        self.dtype = self.data.dtype
        self.shape: Tuple[int] = util.make_tuple(to_dims(self.data.shape) if shape is None else shape)
        self.shape_profile: List = util.make_list(shape)

    def __eq__(self, other) -> bool:
        return self.data == other.data

    def __str__(self) -> str:
        return f"data=({self.data.view()}) : shape=({self.shape}), dtype=({self.dtype.name}), loc=({self.device.kind}:{self.device.index})"

    def to_trace_str(self, input_names, output_names):
        assert not input_names, "Storage should have no inputs!"
        assert len(output_names) == 1, "Storage should have exactly one output!"
        return f"{output_names[0]} : data=({self.data.view()}), shape=({self.shape}), dtype=({self.dtype.name}), stride=(), loc=({self.device.kind}:{self.device.index})"

    def infer_shapes(self, input_shapes):
        assert not input_shapes, "Storage should have no inputs!"
        return [util.make_tuple(self.shape)]

    def infer_dtypes(self, input_dtypes):
        assert not input_dtypes, "Storage should have no inputs!"
        return [self.dtype]

    def infer_devices(self, input_devices):
        assert not input_devices, "Storage should have no inputs!"
        # This is different from self.device
        # Constants are always on device when executed by mlir
        return [make_device("gpu")]

    def to_flat_ir(self, flat_ir, inputs, outputs):
        import cupy as cp

        from tripy.flat_ir.ops import ConstantOp

        assert not inputs, "Storage should have no inputs!"
        data = self.data.view()
        if isinstance(data, cp.ndarray):
            # This is required because MLIR-TRT backend requires constants to be on host.
            data = data.get()
        flat_ir.ops.append(ConstantOp(self, inputs, outputs, data=data))


@TENSOR_METHOD_REGISTRY("__init__")
def tensor_init(
    self: "tripy.Tensor",
    data: Union[List, "np.ndarray", "cp.ndarray", "torch.Tensor", "jnp.ndarray"] = None,
    shape: Optional[ShapeInfo] = None,
    dtype: "tripy.dtype" = None,
    device: "tripy.common.device" = None,
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

        tensor = tp.Tensor([1.0, 2.0, 3.0], shape=(3,), dtype=tp.float32)
    """
    # Note: It is important that we are able to call the Tensor constructor with no arguments
    # since this is used internally by Tensor.build()
    if data is not None:
        from tripy.frontend.ops import Storage

        self._finalize([], Storage(data, to_dims(shape), dtype, device))
