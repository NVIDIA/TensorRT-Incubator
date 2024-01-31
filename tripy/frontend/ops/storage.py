from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union

import tripy.common
from tripy import utils
from tripy.common.array import Array
from tripy.common.types import ShapeInfo
from tripy.frontend.dim import Dim
from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.ops.utils import to_dims


@dataclass
class Storage(BaseOperator):
    """
    Represents data stored in host or device memory.
    """

    data: Array
    shape: ShapeInfo
    dtype: type
    device: tripy.common.device

    def __init__(
        self,
        inputs: List["Tensor"],
        outputs: List["Tensor"],
        const_fold: bool,
        data: Union[List, "np.ndarray", "cp.ndarray", "torch.Tensor", "jnp.ndarray"],
        shape: Optional[Tuple[Dim]] = None,
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
        super().__init__(inputs, outputs, const_fold)

        # Let's not allow user to request a different type unless data is a list.
        if hasattr(data, "to_dlpack"):
            # Ensure that dtype is not set.
            assert dtype is None

        self.device = utils.default(device, tripy.common.device("cpu"))
        self.data = Array(data, dtype, shape, self.device)
        self.dtype = self.data.dtype
        self.shape: ShapeInfo = utils.make_tuple(to_dims(self.data.shape) if shape is None else to_dims(shape))
        self.shape_profile: List = utils.make_list(shape)

    def str_skip_fields(self) -> Set[str]:
        if utils.should_omit_constant_in_str(self.shape):
            return {"data"}
        return set()

    def __eq__(self, other) -> bool:
        return self.data == other.data

    def infer_shapes(self):
        self.outputs[0].shape = self.shape

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def infer_devices(self):
        # This is different from self.device
        # Constants are always on device when executed by mlir
        self.outputs[0].device = tripy.common.device("gpu")

    def to_flat_ir(self, inputs, outputs):
        import cupy as cp

        from tripy.flat_ir.ops import ConstantOp

        data = self.data.view()
        if isinstance(data, cp.ndarray):
            # This is required because MLIR-TRT backend requires constants to be on host.
            data = data.get()
        ConstantOp(self, inputs, outputs, data=data)


@TENSOR_METHOD_REGISTRY("__init__")
def tensor_init(
    self,
    data: Union[List, "np.ndarray", "cp.ndarray", "torch.Tensor", "jnp.ndarray"],
    shape: Optional[ShapeInfo] = None,
    dtype: Optional["tripy.dtype"] = None,
    device: Optional["tripy.device"] = None,
) -> None:
    """
    Creates a tensor.

    Args:
        data: The data with which to initialize the tensor.
        shape: The shape of the tensor.
        dtype: The data type of the tensor.
        device: The device on which to allocate the tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        tensor = tp.Tensor([1.0, 2.0, 3.0], shape=(3,), dtype=tp.float32)
    """
    # Note: It is important that we are able to call the Tensor constructor with no arguments
    # since this is used internally by Tensor.build()
    if data is not None:
        from tripy.frontend.ops import Storage

        self._finalize([], Storage, data, to_dims(shape), dtype, device)
