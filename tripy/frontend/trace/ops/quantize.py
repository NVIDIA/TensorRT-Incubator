from dataclasses import dataclass

from tripy.common import datatype
from tripy.common.exception import raise_error
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Quantize(BaseTraceOp):

    dtype: datatype
    scale: float
    zero_point: int
    storage_min: int
    storage_max: int

    def infer_shapes(self):
        self.outputs[0].shape = self.inputs[0].shape

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import QuantizeOp

        QuantizeOp.build(
            inputs,
            outputs,
            self.scale,
            self.zero_point,
            self.storage_min,
            self.storage_max,
        )


def quantize(
    input: "tripy.Tensor",
    dtype: datatype,
    scale: float,
    zero_point: int,
    storage_min: int,
    storage_max: int,
) -> "tripy.Tensor":
    """
    Quantizes the input Tensor.

    Args:
        input: input Tensor to quantize
        dtype: dtype to quantize
        scale: quantization scale value
        zero_point:
        storage_min:
        storage_max:

    Returns:
        Quantized Tensor.
    """
    from tripy.frontend import Tensor

    if dtype not in (datatype.int8, datatype.int4, datatype.float8e4m3fn):
        raise_error("quantization only supports int4, int8 and float8e4m3fn", [f"Got dtype={dtype}"])

    return Tensor.build([input], Quantize, dtype, scale, zero_point, storage_min, storage_max)
