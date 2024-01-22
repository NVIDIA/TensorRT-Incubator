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
        TANH = 1
        """Perform an elementwise tanh"""

    kind: Kind

    def infer_shapes(self):
        self.outputs[0].shape = self.inputs[0].shape

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ExpOp, TanhOp

        OpType = {
            UnaryElementwise.Kind.EXP: ExpOp,
            UnaryElementwise.Kind.TANH: TanhOp,
        }[self.kind]
        OpType(self, inputs, outputs)


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


def tanh(input: "tripy.Tensor") -> "tripy.Tensor":
    """
    Compute hyperbolic tangent element-wise.

    Returns:
        Corresponding hyperbolic tangent values.

    Example:
    ::

        a = tp.arange(3, dtype=tp.float32)
        out = tp.tanh(a)
        print(out)
        assert np.allclose(out.numpy(), np.tanh(np.arange(3, dtype=np.float32)))
    """
    from tripy.frontend import Tensor

    return Tensor.build([input], UnaryElementwise, UnaryElementwise.Kind.TANH)


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


@TENSOR_METHOD_REGISTRY("tanh")
def _tanh(self: "tripy.Tensor"):
    """
    Equivalent to `tripy.tanh(self)`.
    See 'tripy.tanh'.

    Example:
    ::

        a = tp.arange(3, dtype=tp.float32)
        out = a.tanh()
        print(out)
        assert np.allclose(out.numpy(), np.tanh(np.arange(3, dtype=np.float32)))
    """
    return tanh(self)
