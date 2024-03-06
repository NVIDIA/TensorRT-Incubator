from dataclasses import dataclass

from tripy.common import datatype
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Dequantize(BaseTraceOp):

    dtype: datatype

    def infer_shapes(self):
        self.outputs[0].shape = self.inputs[0].shape

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import DequantizeOp

        DequantizeOp.build(inputs, outputs)


def dequantize(
    input: "tripy.Tensor",
    dtype: datatype,
) -> "tripy.Tensor":
    """
    Quantizes the input Tensor.

    Args:
        input: input quantized Tensor
        dtype: dtype to dequantize

    Returns:
        Dequantized Tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(6, tp.float32).reshape((2, 3))
        quantized = tp.quantize(input, tp.int8, 0.99872)
        output = tp.dequantize(quantized, tp.float32)

        assert np.allclose(output.numpy(), input.numpy(), atol=1e-2)
    """
    from tripy.frontend import Tensor

    # TODO: enable after !215 is merged
    # check if input has a dequantizable dtype
    # if input.dtype not in (datatype.int8, datatype.int4, datatype.float8e4m3fn):
    #     raise_error("input does not have a valid dtype to dequantize", [f"Got dtype={dtype}"])

    return Tensor.build([input], Dequantize, dtype)
