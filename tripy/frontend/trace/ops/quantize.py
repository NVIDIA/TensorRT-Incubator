from dataclasses import dataclass

from tripy.common import datatype
from tripy.common.exception import raise_error
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Quantize(BaseTraceOp):

    dtype: datatype
    scale: float

    def infer_shapes(self):
        self.outputs[0].shape = self.inputs[0].shape

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import QuantizeOp

        def _get_storage_range(dtype):
            # get default range for dtype
            return {
                datatype.int8: (-128, 127),
            }[dtype]

        # 1. tensorrt infers the default storage range
        #    storage_min, storage_max are set to be consistent
        #    with tensorrt, but not really used.
        # 2. tensorrt does not support zero_point, and
        #    mlir-trt requires zero_point == 0
        storage_min, storage_max = _get_storage_range(self.dtype)
        zero_point = 0
        QuantizeOp.build(
            inputs,
            outputs,
            self.scale,
            zero_point,
            storage_min,
            storage_max,
        )


def quantize(
    input: "tripy.Tensor",
    dtype: datatype,
    scale: float,
) -> "tripy.Tensor":
    """
    Quantizes the input Tensor.

    Args:
        input: input Tensor to quantize
        dtype: quantization dtype
        scale: quantization scale value

    Returns:
        Quantized Tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(6, tp.float32).reshape((2, 3))
        quantized = tp.quantize(input, tp.int8, 0.99872)
        output = tp.dequantize(quantized, tp.float32)

        assert np.allclose(output.numpy(), input.numpy(), atol=1e-2)
    """
    from tripy.frontend import Tensor

    # TODO: support other quantization dtypes (int4, fp8 etc)
    if dtype != datatype.int8:
        raise_error("Unsupported quantization dtype.", [f"Supported dtypes: int8, Got dtype={dtype}"])

    return Tensor.build([input], Quantize, dtype, scale)
