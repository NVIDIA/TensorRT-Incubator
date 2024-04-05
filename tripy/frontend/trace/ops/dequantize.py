from dataclasses import dataclass
from typing import Any, Union

from tripy import export
from tripy.common import datatype
from tripy.common.exception import raise_error
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
def dequantize(
    input: "tripy.Tensor",
    scale: Union["tripy.Tensor", Any],
    dtype: datatype.dtype,
    dim: Union[int, Any] = None,
) -> "tripy.Tensor":
    """
    Dequantizes the input tensor.

    If ``dim`` is not given, this function will perform "per-tensor"
    dequantization. The ``scale`` must be a scalar tensor or a single
    python number.

    If ``dim`` is given, this function will perform "per-channel"
    dequantization. The ``scale`` must be a 1-D tensor or a python sequence
    both with size of ``input.shape[dim]``.

    Args:
        input: The input tensor
        scale: The scale tensor
        dtype: Desired data type of the output tensor
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
    """
    from tripy.frontend import Tensor
    from tripy.frontend.trace.ops.unsqueeze import unsqueeze

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
        scale = Tensor(scale if dim is not None else [scale], dtype=dtype)
    elif dim is None:
        # MLIR-TRT needs 1D Tensor in per-tensor mode
        scale = unsqueeze(scale, 0)

    return Tensor.build([input, scale], Dequantize, dtype, dim)
