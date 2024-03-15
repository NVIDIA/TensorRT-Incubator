from dataclasses import dataclass
from typing import Union, Any

from tripy.common import datatype
from tripy.common.exception import raise_error
from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy.utils import export


@dataclass(repr=False)
class Dequantize(BaseTraceOp):

    dtype: datatype.dtype

    def infer_shapes(self):
        self.outputs[0].shape = self.inputs[0].shape

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import DequantizeOp

        DequantizeOp.build(inputs, outputs)


@export.public_api(document_under="tensor")
def dequantize(
    input: "tripy.Tensor",
    scale: Union["tripy.Tensor", Any],
    dtype: datatype.dtype,
) -> "tripy.Tensor":
    """
    Dequantizes the input tensor.

    Args:
        input: The input tensor
        scale: The scale tensor
        dtype: Desired data type of the output tensor

    Returns:
        The dequantized tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(6, tp.int8).reshape((2, 3))
        output = tp.dequantize(input, 0.99872, tp.float32)
    """
    from tripy.frontend import Tensor

    # check if input has a dequantizable dtype
    if input.dtype not in (datatype.int8, datatype.int4, datatype.float8e4m3fn):
        raise_error(
            "Input does not have a valid dtype to dequantize",
            [
                f"input.dtype must be one of `tp.int8, tp.int4, tp.float8e4m3fn`, ",
                f"Got dtype={input.dtype}",
            ],
        )

    if dtype not in (datatype.float32, datatype.float16):
        raise_error(
            "Invalid dequantization dtype.",
            [
                f"dtype must be one of `tp.float32, tp.float16`, ",
                f"Got dtype={dtype}",
            ],
        )

    # TODO(#111): remove this after switching to stablehlo
    if not isinstance(scale, Tensor):
        scale = Tensor([scale], dtype=dtype)

    return Tensor.build([input, scale], Dequantize, dtype)
