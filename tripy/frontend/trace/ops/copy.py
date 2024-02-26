from dataclasses import dataclass

from tripy.common.device import device
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Copy(BaseTraceOp):
    """
    Represents a copy operation.
    """

    target: device

    def infer_shapes(self):
        self.outputs[0].shape = self.inputs[0].shape

    def infer_devices(self):
        self.outputs[0].device = self.target

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import CopyOp

        CopyOp(self, inputs, outputs, target=self.target)


@TENSOR_METHOD_REGISTRY("to")
def to(self, device: "tripy.device") -> "tripy.Tensor":
    r"""
    Returns a copy of this tensor on the target device.

    Args:
        device: The target device.

    Returns:
        A copy of this tensor on target device.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.Tensor([1, 2], device=tp.device("gpu"))
        output = input.to(tp.device("cpu"))

        assert np.array_equal(output.numpy(), np.array([1, 2], dtype=np.float32))
        assert output.op.device.kind == "cpu"
    """
    from tripy.frontend import Tensor
    from tripy.frontend.trace.ops import Storage

    if isinstance(self.op, Storage) and self.op.device == device:
        return self

    return Tensor.build([self], Copy, device)