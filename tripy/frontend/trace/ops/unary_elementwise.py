import enum
from dataclasses import dataclass

from tripy.utils import export
from tripy.frontend.trace.ops.base import BaseTraceOp


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
        from tripy.flat_ir.ops import ExpOp, LogOp, RsqrtOp, TanhOp

        OpType = {
            UnaryElementwise.Kind.EXP: ExpOp,
            UnaryElementwise.Kind.TANH: TanhOp,
            UnaryElementwise.Kind.RSQRT: RsqrtOp,
            UnaryElementwise.Kind.LOG: LogOp,
        }[self.kind]
        OpType.build(inputs, outputs)


@export.public_api(document_under="tensor")
def exp(input: "tripy.Tensor") -> "tripy.Tensor":
    r"""
    Computes the elementwise exponential of the elements of the input tensor:

    :math:`\text{exp}(x_{i}) = e^{x_{i}}`

    Args:
        input: The input tensor.

    Returns:
        A new tensor of the same shape and data type as the input tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(3, dtype=tp.float32)
        output = tp.exp(input)

        assert np.allclose(output.numpy(), np.exp(input.numpy()))
    """
    from tripy.frontend import Tensor

    return Tensor.build([input], UnaryElementwise, UnaryElementwise.Kind.EXP)


@export.public_api(document_under="tensor")
def tanh(input: "tripy.Tensor") -> "tripy.Tensor":
    """
    Computes the elementwise hyperbolic tangent of the elements of the input tensor.

    Args:
        input: The input tensor.

    Returns:
        A new tensor of the same shape and data type as the input tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(3, dtype=tp.float32)
        output = tp.tanh(input)

        assert np.allclose(output.numpy(), np.tanh(input.numpy()))
    """
    from tripy.frontend import Tensor

    return Tensor.build([input], UnaryElementwise, UnaryElementwise.Kind.TANH)


@export.public_api(document_under="tensor")
def rsqrt(input: "tripy.Tensor") -> "tripy.Tensor":
    """
    Computes the elementwise reciprocal square root of the elements of the input tensor.

    Args:
        input: The input tensor.

    Returns:
        A new tensor of the same shape and data type as the input tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(3, dtype=tp.float32) + 1.0
        output = tp.rsqrt(input)

        assert np.allclose(output.numpy(), (1.0 / np.sqrt(input.numpy())))
    """
    from tripy.frontend import Tensor

    return Tensor.build([input], UnaryElementwise, UnaryElementwise.Kind.RSQRT)


@export.public_api(document_under="tensor")
def log(input: "tripy.Tensor") -> "tripy.Tensor":
    """
    Computes the elementwise natural logarithm (base e) of the elements of the input tensor.

    Args:
        input: The input tensor.

    Returns:
        A new tensor of the same shape and data type as the input tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(1, 3, dtype=tp.float32)
        output = tp.log(input)

        assert np.allclose(output.numpy(), (np.log(input.numpy())))
    """
    from tripy.frontend import Tensor

    return Tensor.build([input], UnaryElementwise, UnaryElementwise.Kind.LOG)
