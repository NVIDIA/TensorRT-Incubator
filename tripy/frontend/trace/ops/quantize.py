from dataclasses import dataclass
from typing import Any, Union

from tripy import export
from tripy.common import datatype
from tripy.common.exception import raise_error
from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy.frontend import utils as frontend_utils


@dataclass(repr=False)
class Quantize(BaseTraceOp):

    dtype: datatype.dtype
    dim: int

    def infer_shapes(self):
        self.outputs[0].shape = self.inputs[0].shape

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import QuantizeOp

        QuantizeOp.build(inputs, outputs, self.dim)


@export.public_api(document_under="tensor_operations")
# TODO(#111): remove this after switching to stablehlo
@frontend_utils.convert_inputs_to_tensors(exclude=["dtype", "dim"])
def quantize(
    input: "tripy.Tensor",
    scale: Union["tripy.Tensor", Any],
    dtype: datatype.dtype,
    dim: Union[int, Any] = None,
) -> "tripy.Tensor":
    """
    Quantizes the input Tensor. The valid quantized data types are
    :class:`tripy.int8`, :class:`tripy.int4`, :class:`tripy.float8`.

    If ``dim`` is not given, this function will perform "per-tensor"
    quantization. The ``scale`` must be a scalar tensor or a single
    python number.

    If ``dim`` is given, this function will perform "per-channel"
    quantization. The ``scale`` must be a 1-D tensor or a python sequence
    both with size of ``input.shape[dim]``.

    Args:
        input: The input tensor with data type of :class:`float32` or :class:`float16`.
        scale: The scale tensor
        dtype: The quantization data type. Must be a valid quantized data type (see above).
        dim: The dimension for per-channel quantization

    Returns:
        Quantized Tensor.

    .. code-block:: python
        :linenos:
        :caption: Per-tensor quantization

        input = tp.reshape(tp.arange(6, tp.float32), (2, 3))
        scale = 0.99872
        output = tp.quantize(input, scale, tp.int8)

        expected = (np.reshape(np.arange(6, dtype=np.float32), (2, 3)) / scale).astype(np.int8) # doc: omit
        assert np.array_equal(output.numpy(), expected)

    .. code-block:: python
        :linenos:
        :caption: Per-channel quantization

        input = tp.Tensor([[0, 1, 2], [3, 4, 5]], dtype=tp.float32)
        scale = [0.99872, 0.96125]
        output = tp.quantize(input, scale, tp.int8, dim=0)

        expected = (np.reshape(np.arange(6, dtype=np.float32), (2, 3)) / np.array(scale).reshape(2, 1)).astype(np.int8) # doc: omit
        assert np.array_equal(output.numpy(), expected)

    .. seealso:: :func:`dequantize`
    """
    from tripy.frontend.trace.ops.cast import cast
    from tripy.logging import logger

    if input.dtype not in (datatype.float32, datatype.float16):
        raise_error(
            "Input does not have a valid dtype to quantize.",
            [
                f"input.dtype must be one of `tp.float32, tp.float16`, ",
                f"Got dtype={input.dtype}",
            ],
        )

    # TODO(#164): support int4 quantization dtype
    SUPPORTED_QUANT_DTYPES = [datatype.int8, datatype.float8]
    if dtype not in (datatype.int8, datatype.float8):
        raise_error(
            "Unsupported quantization dtype.",
            [
                f"Supported dtypes are: {SUPPORTED_QUANT_DTYPES}.",
                f"Got dtype={dtype}",
            ],
        )

    # TODO(MLIR-TRT #770) Remove scale casting
    if scale.dtype != datatype.float32:
        logger.warning(f"Casting scale to `tripy.float32`, original dtype is {scale.dtype}.")
        scale = cast(scale, datatype.float32)

    return Quantize.build([input, scale], dtype, dim)
