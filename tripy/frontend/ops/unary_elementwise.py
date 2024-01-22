import enum
from dataclasses import dataclass

from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY


@dataclass
class UnaryElementwise(BaseOperator):
    """
    Represents a unary elementwise operation.
    """

    class Kind(enum.Enum):
        EXP = 0
        """Perform an elementwise exponential"""

    kind: Kind

    def infer_shapes(self):
        self.outputs[0].shape = self.inputs[0].shape

    def to_flat_ir(self, flat_ir):
        from tripy.flat_ir.ops import ExpOp

        if self.kind == UnaryElementwise.Kind.EXP:
            flat_ir.add_op(self, ExpOp, self.inputs, self.outputs)
        else:
            raise NotImplementedError()


def exp(input: "tripy.Tensor"):
    """
    Returns a tensor with the exponential of the elements of the input tensor

    Returns:
        The output Tensor.

    Example:
    ::

        a = tp.arange(3, dtype=tp.float32)
        out = tp.exp(a)
        print(out)
        assert np.allclose(out.numpy(), np.exp(np.arange(3, dtype=np.float32)))
    """
    from tripy.frontend import Tensor

    return Tensor.build([input], UnaryElementwise, UnaryElementwise.Kind.EXP)


@TENSOR_METHOD_REGISTRY("exp")
def _exp(self: "tripy.Tensor"):
    """
    Equivalent to `tripy.exp(self)`.
    See 'tripy.exp'.

    Example:
    ::

        a = tp.arange(3, dtype=tp.float32)
        out = a.exp()
        print(out)
        assert np.allclose(out.numpy(), np.exp(np.arange(3, dtype=np.float32)))
    """
    return exp(self)
