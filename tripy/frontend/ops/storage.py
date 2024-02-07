from dataclasses import dataclass
from typing import List, Set

import tripy.common
from tripy import utils
from tripy.common.array import Array
from tripy.frontend.ops.base import BaseOperator
from tripy.common.types import ShapeInfo


@dataclass(repr=False)
class Storage(BaseOperator):
    """
    Represents data stored in host or device memory.
    """

    data: Array
    shape: ShapeInfo  # This is a ShapeInfo but will always be a static shape
    dtype: type
    device: tripy.common.device

    def __init__(self, inputs: List["Tensor"], outputs: List["Tensor"], data: Array) -> None:
        super().__init__(inputs, outputs)

        self.data = data
        self.shape = utils.to_dims(data.shape)
        self.dtype = data.dtype
        self.device = data.device

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
