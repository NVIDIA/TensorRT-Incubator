from dataclasses import dataclass
from typing import List

from tripy.common.device import device
from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.util import make_tuple


@dataclass
class Copy(BaseOperator):
    """
    Represents a copy operation.
    """

    target: device

    def to_trace_str(self, input_names, output_names):
        assert len(input_names) == 1, "Copy should have exactly one input!"
        assert len(output_names) == 1, "Copy should have exactly one output!"
        return f"{output_names[0]} = copy({input_names[0]}, target = {self.target.kind}:{self.target.index})"

    def infer_shapes(self, input_shapes):
        return [make_tuple(input_shapes[0])]

    def infer_dtypes(self, input_dtypes):
        return [input_dtypes[0]]

    def infer_devices(self, input_devices: List) -> List:
        return [self.target]

    def to_flat_ir(self, flat_ir, inputs, outputs):
        from tripy.flat_ir.ops import CopyOp

        flat_ir.ops.append(CopyOp(self, inputs, outputs, target=self.target))


@TENSOR_METHOD_REGISTRY("to")
def to(self: "tripy.Tensor", device: "tripy.device"):
    """
    Copies input Tensor to the target device.

    Args:
        device: target device

    Returns:
        Copied Tensor on target device

    Example:
    ::

        import numpy as np

        a = tp.Tensor([1, 2], device=tp.device("gpu"))
        a = a.to(tp.device("cpu"))
        assert (a.numpy() == np.array([1, 2], dtype=np.float32)).all()
        assert a.op.device.kind == "cpu"
    """
    from tripy.frontend import Tensor
    from tripy.frontend.ops import Storage

    if isinstance(self.op, Storage) and self.op.device == device:
        return self

    return Tensor.build([self], Copy, device)
