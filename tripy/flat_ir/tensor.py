import dataclasses
from dataclasses import dataclass

from mlir import ir

from tripy.frontend import Dim
from tripy.frontend.trace.tensor import TraceTensor
from tripy.util import make_list


@dataclass
class FIRTensor(TraceTensor):
    """
    Represents a single tensor in the FlatIR
    """

    def __init__(self, instance):
        super().__init__(**{field.name: getattr(instance, field.name) for field in dataclasses.fields(instance)})

    def to_mlir(self):
        from tripy.backend.mlir import utils as mlir_utils

        return ir.RankedTensorType.get(
            [ir.ShapedType.get_dynamic_size() if s.is_dynamic_dim() else s.min for s in make_list(self.shape)],
            mlir_utils.get_mlir_dtype(self.dtype),
        )
