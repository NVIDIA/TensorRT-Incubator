from dataclasses import dataclass
from typing import Any, Union

from tripy import export
from tripy.common import datatype
from tripy.common.exception import raise_error
from tripy.frontend import utils as frontend_utils
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Dequantize(BaseTraceOp):

    dtype: datatype.dtype
    dim: int

    def infer_shapes(self):
        self.outputs[0].shape = self.inputs[0].shape

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import DequantizeOp

        DequantizeOp.build(inputs, outputs, self.dim)


@export.public_api(document_under="tensor_operations")
# TODO(#111): remove this after switching to stablehlo
@frontend_utils.convert_inputs_to_tensors(exclude=["dtype", "dim"])
def dequantize(
    input: "tripy.Tensor",
    scale: Union["tripy.Tensor", Any],
    dtype: datatype.dtype,
    dim: Union[int, Any] = None,
) -> "tripy.Tensor":
    """
    Dequantizes the input tensor.

    If ``dim`` is not given, this function will perform "per-tensor"
    or "block-wise" dequantization.

    * For "per-tensor" dequantization, the ``scale`` must be a scalar
      tensor or a single python number.

    * For "block-wise" dequantization, the ``dtype`` must only be :class:`tripy.int4`.
      The ``input`` tensor must only have 2 dimensions, e.g. ``[D0, D1]``.
      The ``scale`` must also be a 2-D tensor or a 2-D python sequence.
      The first dimension of ``scale`` must be able to divide ``D0``,
      where "blocking" is performed. The second dimension of ``scale``
      must equal to ``D1``.


    If ``dim`` is given, this function will perform "per-channel"
    dequantization. The ``scale`` must be a 1-D tensor or a python sequence
    both with size of ``input.shape[dim]``.

    Args:
        input: The input tensor with a valid quantized data type.
        scale: The scale tensor
        dtype: The data type after dequantization. Must be :class:`tripy.float32` or :class:`tripy.float16`.
        dim: The dimension for per-channel dequantization

    Returns:
        The dequantized tensor.

    .. code-block:: python
        :linenos:
        :caption: Per-tensor dequantization

        input = tp.Tensor([1, 2, 3], dtype=tp.int8)
        scale = 0.99872
        output = tp.dequantize(input, scale, tp.float32)

        expected = (np.array([1, 2, 3], dtype=np.int8) * scale).astype(np.float32) # doc: omit
        assert np.array_equal(output.numpy(), expected)

    .. code-block:: python
        :linenos:
        :caption: Per-channel dequantization

        input = tp.Tensor([[1, 2, 3], [4, 5, 6]], dtype=tp.int8)
        scale = [0.99872, 0.96125]
        output = tp.dequantize(input, scale, tp.float32, dim=0)

        expected = (np.array([[1, 2, 3], [4, 5, 6]]) * np.array(scale).reshape(2, 1)).astype(np.float32) # doc: omit
        assert np.array_equal(output.numpy(), expected)

    .. code-block:: python
        :linenos:
        :caption: Block-wise dequantization

        # doc: print-locals input, output

        input = tp.Tensor([[0, 1], [2, 3]], dtype=tp.float32)
        scale = [[1.0, 1.0]]
        quant = tp.quantize(input, scale, tp.int4)
        output = tp.dequantize(quant, scale, tp.float32)

        assert np.array_equal(output.numpy(), np.array([[0, 1], [2, 3]], dtype=np.float32))

    .. seealso:: :func:`quantize`
    """
    from tripy.frontend.trace.ops.cast import cast
    from tripy.logging import logger

    # check if input has a dequantizable dtype
    VALID_DEQUANT_DTYPES = [datatype.int8, datatype.int4, datatype.float8]
    if input.dtype not in VALID_DEQUANT_DTYPES:
        raise_error(
            "Input does not have a valid dtype to dequantize",
            [
                f"input.dtype must be one of {VALID_DEQUANT_DTYPES}.",
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

    # TODO(MLIR-TRT #770) Remove scale casting
    if scale.dtype != datatype.float32:
        logger.warning(f"Casting scale to `tripy.float32`, original dtype is {scale.dtype}.")
        scale = cast(scale, datatype.float32)

    return Dequantize.build([input, scale], dtype, dim)
