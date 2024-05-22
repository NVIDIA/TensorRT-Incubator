from typing import List, Union
from dataclasses import dataclass

from tripy import export
from tripy.common.device import device
from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy.frontend.utils import convert_inputs_to_tensors
from .reshape import reshape

from tripy import utils


@dataclass(repr=False)
class Concatenate(BaseTraceOp):
    dim: int

    def infer_shapes(self):
        # don't add check for non dim shapes matching since with DS the check should happen inside mlir-tensorrt.

        self.dim = self.dim + len(self.inputs[0].shape) if self.dim < 0 else self.dim
        concat_dim = sum(inp.shape[self.dim].runtime_value for inp in self.inputs)
        self.outputs[0].shape = utils.to_dims(
            [s if idx != self.dim else concat_dim for idx, s in enumerate(self.inputs[0].shape)]
        )

    def infer_devices(self):
        self.outputs[0].device = self.inputs[0].device

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ConcatenateOp

        ConcatenateOp.build(inputs, outputs, dim=self.dim)


@export.public_api(document_under="tensor_operations")
def concatenate(tensors: List[Union["tripy.Tensor"]], dim: int) -> "tripy.Tensor":
    r"""
    Returns a copy of the input tensor on the target device.

    Args:
        tensors: List of tensors of the same type and having the same shape except in the concatenated dimension.
        dim: the dimension over which the tensors are concatenated.

    Returns:
        Concatenated tensor with shape along `dim` axis equal to sum of dimensions at `dim` axis for all inputs.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.iota((2, 3), dtype=tp.float32)
        b = tp.iota((4, 3), dtype=tp.float32)

        output = tp.concatenate([a, b], dim=0)

        assert np.array_equal(cp.from_dlpack(output).get(), np.concatenate((cp.from_dlpack(a).get(), cp.from_dlpack(b).get()), axis=0))
    """
    return Concatenate.build(tensors, dim)
