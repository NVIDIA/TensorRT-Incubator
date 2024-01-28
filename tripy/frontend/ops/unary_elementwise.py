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

        input = tp.arange(3, dtype=tp.float32)
        output = input.exp()

        assert np.allclose(output.numpy(), np.exp(input.numpy()))
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

        input = tp.arange(3, dtype=tp.float32)
        output = input.tanh()

        assert np.allclose(output.numpy(), np.tanh(input.numpy()))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self], UnaryElementwise, UnaryElementwise.Kind.TANH)


@TENSOR_METHOD_REGISTRY("rsqrt")
def rsqrt(self: "tripy.Tensor"):
    """
    Compute reciprocal square root operation on tensor.

    Example:

    .. code:: python

        input = tp.arange(3, dtype=tp.float32)
        output = input.rsqrt()

        assert np.allclose(output.numpy(), (1.0 / np.sqrt(input.numpy())))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self], UnaryElementwise, UnaryElementwise.Kind.RSQRT)
