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


@TENSOR_METHOD_REGISTRY("exp")
def exp(self: "tripy.Tensor") -> "tripy.Tensor":
    r"""
    Performs an exponential of the elements of the input tensor:

    :math:`\text{exp}(x_{i}) = e^{x_{i}}`

    Returns:
        Exponential of the input.

    Example:
    ::

        a = tp.arange(3, dtype=tp.float32)
        out = a.exp()
        print(out)
        assert np.allclose(out.numpy(), np.exp(np.arange(3, dtype=np.float32)))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self], UnaryElementwise, UnaryElementwise.Kind.EXP)


@TENSOR_METHOD_REGISTRY("tanh")
def tanh(self: "tripy.Tensor") -> "tripy.Tensor":
    """
    Compute hyperbolic tangent element-wise.

    Returns:
        Hyperbolic tangent values.

    Example:
    ::

        a = tp.arange(3, dtype=tp.float32)
        out = a.tanh()
        print(out)
        assert np.allclose(out.numpy(), np.tanh(np.arange(3, dtype=np.float32)))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self], UnaryElementwise, UnaryElementwise.Kind.TANH)
