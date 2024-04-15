from dataclasses import dataclass

from tripy import export
from tripy.common.device import device
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Copy(BaseTraceOp):
    target: device

    def infer_shapes(self):
        self.outputs[0].shape = self.inputs[0].shape

    def infer_devices(self):
        self.outputs[0].device = self.target

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import CopyOp

        CopyOp.build(inputs, outputs, target=self.target)


@export.public_api(document_under="tensor_operations")
def copy(input: "tripy.Tensor", device: "tripy.device") -> "tripy.Tensor":
    r"""
    Returns a copy of the input tensor on the target device.

    Args:
        input:
        device: The target device.

    Returns:
        A copy of this tensor on target device.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.Tensor([1, 2], device=tp.device("gpu"))
        output = tp.copy(input, tp.device("cpu"))

        assert np.array_equal(output.numpy(), np.array([1, 2], dtype=np.float32))
        assert output.trace_tensor.producer.device.kind == "cpu"
    """
    from tripy.frontend import Tensor
    from tripy.frontend.trace.ops import Storage

    if isinstance(input.trace_tensor.producer, Storage) and input.trace_tensor.producer.device == device:
        return input

    return Tensor.build([input], Copy, device)
