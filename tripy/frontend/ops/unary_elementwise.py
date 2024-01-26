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
        TANH = 1
        RSQRT = 2

    kind: Kind

    def infer_shapes(self):
        self.outputs[0].shape = self.inputs[0].shape

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ExpOp, TanhOp, RsqrtOp

        OpType = {
            UnaryElementwise.Kind.EXP: ExpOp,
            UnaryElementwise.Kind.TANH: TanhOp,
            UnaryElementwise.Kind.RSQRT: RsqrtOp,
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

    .. code:: python
        :number-lines:

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

    .. code:: python
        :number-lines:

        a = tp.arange(3, dtype=tp.float32)
        out = a.tanh()
        print(out)
        assert np.allclose(out.numpy(), np.tanh(np.arange(3, dtype=np.float32)))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self], UnaryElementwise, UnaryElementwise.Kind.TANH)


@TENSOR_METHOD_REGISTRY("rsqrt")
def rsqrt(self: "tripy.Tensor"):
    """
    Compute reciprocal square root operation on tensor.

    Example:

    .. code:: python
        :number-lines:

        a = tp.arange(3, dtype=tp.float32)
        out = a.rsqrt()
        print(out)
        assert np.allclose(out.numpy(), (1.0 / np.sqrt(np.arange(3, dtype=np.float32))))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self], UnaryElementwise, UnaryElementwise.Kind.RSQRT)
