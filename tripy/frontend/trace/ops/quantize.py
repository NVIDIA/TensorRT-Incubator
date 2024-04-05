from dataclasses import dataclass
from typing import Any, Union

from tripy import export
from tripy.common import datatype
from tripy.common.exception import raise_error
from tripy.frontend.trace.ops.base import BaseTraceOp


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
def quantize(
    input: "tripy.Tensor",
    scale: Union["tripy.Tensor", Any],
    dtype: datatype.dtype,
    dim: Union[int, Any] = None,
) -> "tripy.Tensor":
    """
    Quantizes the input Tensor.

    If ``dim`` is not given, this function will perform "per-tensor"
    quantization. The ``scale`` must be a scalar tensor or a single
    python number.

    If ``dim`` is given, this function will perform "per-channel"
    quantization. The ``scale`` must be a 1-D tensor or a python sequence
    both with size of ``input.shape[dim]``.

    Args:
        input: The input tensor
        scale: The scale tensor
        dtype: Desired data type of the output tensor
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
    """
    from tripy.frontend import Tensor
    from tripy.frontend.trace.ops.unsqueeze import unsqueeze

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
        scale = Tensor(scale if dim is not None else [scale], dtype=input.dtype)
    elif dim is None:
        # MLIR-TRT needs 1D Tensor in per-tensor mode
        scale = unsqueeze(scale, 0)

    return Tensor.build([input, scale], Quantize, dtype, dim)
