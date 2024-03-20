from dataclasses import dataclass

from tripy import export
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Cast(BaseTraceOp):
    to_type: "tripy.common.dtype"

    def infer_shapes(self):
        self.outputs[0].shape = self.inputs[0].shape

    def infer_dtypes(self):
        self.outputs[0].dtype = self.to_type

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ConvertOp

        ConvertOp.build(inputs, outputs)


@export.public_api(document_under="tensor_operations")
def cast(input: "tripy.Tensor", dtype: "tripy.dtype") -> "tripy.Tensor":
    r"""
    Returns a tensor with the contents of the input tensor casted to the specified data type.

    Args:
        input: The input tensor.
        dtype: The desired data type.

    Returns:
        A tensor containing the casted values.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.Tensor([1, 2], dtype=tp.int32)
        output = tp.cast(input, tp.float32)

        assert np.array_equal(output.numpy(), np.array([1, 2], dtype=np.float32))
    """
    from tripy.frontend import Tensor

    if input.dtype == dtype:
        return input

    return Tensor.build([input], Cast, dtype)
