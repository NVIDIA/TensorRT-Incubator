from dataclasses import dataclass

from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY


@dataclass
class Cast(BaseOperator):
    """
    Represents a cast operation.
    """

    to_type: "tripy.common.dtype"

    def infer_shapes(self):
        self.outputs[0].shape = self.inputs[0].shape

    def infer_dtypes(self):
        self.outputs[0].dtype = self.to_type

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ConvertOp

        ConvertOp(self, inputs, outputs)


@TENSOR_METHOD_REGISTRY("to")
def to(self: "tripy.Tensor", dtype: "tripy.dtype") -> "tripy.Tensor":
    r"""
    Returns a tensor with the specified data type.

    Args:
        dtype: The target data type.

    Returns:
        The casted tensor.

    Example:

    .. code:: python
        :number-lines:

        inp = tp.Tensor([1, 2], dtype=tp.int32)
        print(f"inp: {inp}")
        out = inp.to(tp.float32)
        print(f"out: {out}")

        assert np.array_equal(out.numpy(), np.array([1, 2], dtype=np.float32))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self], Cast, dtype)
