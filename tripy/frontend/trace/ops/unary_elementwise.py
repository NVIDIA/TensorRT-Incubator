import enum
from dataclasses import dataclass

from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY


@dataclass(repr=False)
class UnaryElementwise(BaseTraceOp):
    class Kind(enum.Enum):
        EXP = 0
        TANH = 1
        RSQRT = 2
        LOG = 3

    kind: Kind

    def infer_shapes(self):
        self.outputs[0].shape = self.inputs[0].shape

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ExpOp, TanhOp, RsqrtOp, LogOp

        OpType = {
            UnaryElementwise.Kind.EXP: ExpOp,
            UnaryElementwise.Kind.TANH: TanhOp,
            UnaryElementwise.Kind.RSQRT: RsqrtOp,
            UnaryElementwise.Kind.LOG: LogOp,
        }[self.kind]
        OpType.build(inputs, outputs)


@TENSOR_METHOD_REGISTRY("exp")
def exp(self) -> "tripy.Tensor":
    r"""
    Computes the elementwise exponential of the elements of this tensor:

    :math:`\text{exp}(x_{i}) = e^{x_{i}}`

    Returns:
        A new tensor of the same shape and data type as this tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(3, dtype=tp.float32)
        output = input.exp()

        assert np.allclose(output.numpy(), np.exp(input.numpy()))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self], UnaryElementwise, UnaryElementwise.Kind.EXP)


@TENSOR_METHOD_REGISTRY("tanh")
def tanh(self) -> "tripy.Tensor":
    """
    Computes the elementwise hyperbolic tangent of the elements of this tensor.

    Returns:
        A new tensor of the same shape and data type as this tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(3, dtype=tp.float32)
        output = input.tanh()

        assert np.allclose(output.numpy(), np.tanh(input.numpy()))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self], UnaryElementwise, UnaryElementwise.Kind.TANH)


@TENSOR_METHOD_REGISTRY("rsqrt")
def rsqrt(self) -> "tripy.Tensor":
    """
    Computes the elementwise reciprocal square root of the elements of this tensor.

    Returns:
        A new tensor of the same shape and data type as this tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(3, dtype=tp.float32) + 1.0
        output = input.rsqrt()

        assert np.allclose(output.numpy(), (1.0 / np.sqrt(input.numpy())))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self], UnaryElementwise, UnaryElementwise.Kind.RSQRT)


@TENSOR_METHOD_REGISTRY("log")
def log(self) -> "tripy.Tensor":
    """
    Computes the elementwise natural logarithm (base e) of the elements of this tensor.

    Returns:
        A new tensor of the same shape and data type as this tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(1, 3, dtype=tp.float32)
        output = input.log()

        assert np.allclose(output.numpy(), (np.log(input.numpy())))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self], UnaryElementwise, UnaryElementwise.Kind.LOG)
