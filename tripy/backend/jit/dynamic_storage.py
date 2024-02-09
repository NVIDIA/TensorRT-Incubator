from dataclasses import dataclass
from typing import List, Optional

from tripy import utils
from tripy.common.array import Array
from tripy.common.types import ShapeInfo
from tripy.frontend.trace.ops import Storage


@dataclass(repr=False)
class DynamicStorage(Storage):
    def __init__(
        self, inputs: List["Tensor"], outputs: List["Tensor"], data: Array, dynamic_shape: Optional[ShapeInfo]
    ):
        super().__init__(inputs, outputs, data)
        # Potentially replace the shape with a dynamic one.
        self.shape = utils.default(dynamic_shape, self.shape)
