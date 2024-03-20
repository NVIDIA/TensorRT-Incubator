from dataclasses import dataclass
from typing import Any, Union

from tripy import export
from tripy.common import datatype
from tripy.common.exception import raise_error
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Quantize(BaseTraceOp):

    dtype: datatype.dtype

    def infer_shapes(self):
        self.outputs[0].shape = self.inputs[0].shape

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import QuantizeOp

        QuantizeOp.build(inputs, outputs)


@export.public_api(document_under="tensor_operations")
def quantize(
    input: "tripy.Tensor",
    scale: Union["tripy.Tensor", Any],
    dtype: datatype.dtype,
) -> "tripy.Tensor":
    """
    Quantizes the input Tensor.

    Args:
        input: input Tensor to quantize
        scale: Tensor that contains the scale value
        dtype: quantization dtype

    Returns:
        Quantized Tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.reshape(tp.arange(6, tp.float32), (2, 3))
        scale = 0.99872
        output = tp.quantize(input, scale, tp.int8)

        expected = (np.reshape(np.arange(6, dtype=np.float32), (2, 3)) / scale).astype(np.int8) # doc: omit
        assert np.array_equal(output.numpy(), expected)
    """
    from tripy.frontend import Tensor

    if input.dtype not in (datatype.float32, datatype.float16):
        raise_error(
            "Input does not have a valid dtype to quantize.",
            [
                f"input.dtype must be one of `tp.float32, tp.float16`, ",
                f"Got dtype={input.dtype}",
            ],
        )

    # TODO: support other quantization dtypes (int4, fp8 etc)
    if dtype != datatype.int8:
        raise_error("Unsupported quantization dtype.", [f"Supported dtypes: `tp.int8`, Got dtype={dtype}"])

    # TODO(#111): remove this after switching to stablehlo
    if not isinstance(scale, Tensor):
        scale = Tensor([scale], dtype=input.dtype)

    return Tensor.build([input, scale], Quantize, dtype)
