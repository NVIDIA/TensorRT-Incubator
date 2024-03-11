import enum
from dataclasses import dataclass
from typing import Optional, Sequence, Union

from tripy import utils
from tripy.common import datatype
from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy.utils import export, make_list


@dataclass(repr=False)
class Reduce(BaseTraceOp):
    class Kind(enum.Enum):
        def __init__(self, op, init_value):
            self.op = op
            self.init_value = init_value

        SUM = "sum", 0
        MAX = "max", 0
        MUL = "mul", 1

    dim: Sequence[int]
    kind: Kind

    def infer_shapes(self):
        input_shape = self.inputs[0].shape

        if self.dim is None:
            out_shape = []
            self.dim = list(range(len(input_shape)))
        else:
            self.dim = make_list(self.dim)
            self.dim = [idx if idx >= 0 else idx + len(input_shape) for idx in self.dim]
            out_shape = []
            for idx, s in enumerate(input_shape):
                if idx not in self.dim:
                    out_shape.append(s)
        self.outputs[0].shape = utils.to_dims(out_shape)

    def to_flat_ir(self, inputs, outputs):
        import numpy as np

        from tripy.common.array import Array
        from tripy.common.device import device
        from tripy.flat_ir.ops import ConstantOp, ReduceOp
        from tripy.flat_ir.tensor import FlatIRTensor

        init_value = self.kind.init_value
        init_const = FlatIRTensor.build(shape=[], dtype=outputs[0].dtype, device=outputs[0].device)
        ConstantOp.build(
            [],
            [init_const],
            data=Array(
                np.array(init_value, dtype=outputs[0].dtype.name), outputs[0].dtype, shape=(), device=device("cpu")
            ),
        )
        ReduceOp.build([inputs[0], init_const], outputs, reduce_mode=self.kind.op, reduce_dims=self.dim)


@dataclass(repr=False)
class ArgMinMax(Reduce):

    class Kind:
        ARG_MAX = "argmax"
        ARG_MIN = "argmin"

    dim: Sequence[int]
    kind: Kind

    def infer_dtypes(self):
        self.outputs[0].dtype = datatype.int32

    def to_flat_ir(self, inputs, outputs):
        import numpy as np

        from tripy.common.array import Array
        from tripy.common.device import device
        from tripy.flat_ir.ops import ArgMinMaxOp, ConstantOp
        from tripy.flat_ir.tensor import FlatIRTensor

        init_val_const = FlatIRTensor.build(shape=[], dtype=inputs[0].dtype, device=outputs[0].device)
        init_idx_const = FlatIRTensor.build(shape=[], dtype=outputs[0].dtype, device=outputs[0].device)

        ConstantOp.build(
            [],
            [init_val_const],
            data=Array(np.array(0, dtype=inputs[0].dtype.name), inputs[0].dtype, shape=(), device=device("cpu")),
        )
        ConstantOp.build(
            [],
            [init_idx_const],
            data=Array(np.array(0, dtype=outputs[0].dtype.name), outputs[0].dtype, shape=(), device=device("cpu")),
        )

        ArgMinMaxOp.build(
            [inputs[0], inputs[1], init_val_const, init_idx_const],
            outputs,
            reduce_mode=self.kind,
            reduce_dims=self.dim,
        )


def _reduce_impl(self, kind: Reduce.Kind, dim: Union[int, Sequence], keepdim: bool):
    from tripy.frontend.tensor import Tensor
    from tripy.frontend.trace.ops.unsqueeze import unsqueeze

    out = Tensor.build([self], Reduce, dim, kind)
    if keepdim:
        if dim is None:
            # TODO(#96): Support dim=None, keepdim=True
            raise NotImplementedError("dim=None, keepdim=True is not supported yet.")
        for d in sorted(make_list(dim)):
            out = unsqueeze(out, d)

    return out


@export.public_api(document_under="tensor/reduction_operations")
def sum(
    input: "tripy.Tensor", dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False
) -> "tripy.Tensor":
    """
    Returns a new tensor containing the sum of the elements of the input tensor along the specified dimension.

    Args:
        input: The input tensor.
        dim: The dimension or dimensions along which to reduce.
            If this is not provided, all dimensions are reduced.
        keepdim: Whether to retain reduced dimensions in the output.
            If this is False, reduced dimensions will be squeezed.

    Returns:
        A new tensor of the same data type as the input tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (2, 3))
        output = tp.sum(input, 0)

        assert np.array_equal(output.numpy(), np.sum(np.arange(6, dtype=np.float32).reshape((2, 3)), 0))
    """
    return _reduce_impl(input, Reduce.Kind.SUM, dim, keepdim)


@export.public_api(document_under="tensor/reduction_operations")
def max(
    input: "tripy.Tensor", dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False
) -> "tripy.Tensor":
    """
    Returns a new tensor containing the maximum of the elements of the input tensor along the specified dimension.

    Args:
        input: The input tensor.
        dim: The dimension or dimensions along which to reduce.
            If this is not provided, all dimensions are reduced.
        keepdim: Whether to retain reduced dimensions in the output.
            If this is False, reduced dimensions will be squeezed.

    Returns:
        A new tensor of the same data type as the input tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (2, 3))
        output = tp.max(input, 0)

        assert np.array_equal(output.numpy(), np.max(np.arange(6, dtype=np.float32).reshape((2, 3)), 0))
    """
    return _reduce_impl(input, Reduce.Kind.MAX, dim, keepdim)


@export.public_api(document_under="tensor/reduction_operations")
def prod(
    input: "tripy.Tensor", dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False
) -> "tripy.Tensor":
    """
    Returns a new tensor containing the product of the elements of the input tensor along the specified dimension.

    Args:
        input: The input tensor.
        dim: The dimension or dimensions along which to reduce.
            If this is not provided, all dimensions are reduced.
        keepdim: Whether to retain reduced dimensions in the output.
            If this is False, reduced dimensions will be squeezed.

    Returns:
        A new tensor of the same data type as the input tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (2, 3))
        output = tp.prod(input, 0)

        assert np.array_equal(output.numpy(), np.prod(np.arange(6, dtype=np.float32).reshape((2, 3)), 0))
    """
    return _reduce_impl(input, Reduce.Kind.MUL, dim, keepdim)


def mean_impl(tensor: "tripy.Tensor", dim: Union[int, Sequence] = None, keepdim: bool = False, apply_to_divisor=None):
    from tripy.frontend.trace.ops.cast import cast

    sum_val = sum(tensor, dim=dim, keepdim=keepdim)
    # compute number of elements in the array and divide by number of elements in dims
    input_shape = tensor.shape
    nb_elements = prod(input_shape, dim=0, keepdim=True)
    nb_elements_in_mean_dim = 1

    if dim is not None:
        for d in make_list(dim):
            nb_elements_in_mean_dim = input_shape[d] * nb_elements_in_mean_dim
        divisor = nb_elements_in_mean_dim
    else:
        divisor = nb_elements

    if apply_to_divisor:
        divisor = apply_to_divisor(divisor)

    return sum_val / (cast(divisor, sum_val.dtype))


@export.public_api(document_under="tensor/reduction_operations")
def mean(
    input: "tripy.Tensor", dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False
) -> "tripy.Tensor":
    """
    Returns a new tensor containing the mean of the elements of the input tensor along the specified dimension.

    Args:
        input: The input tensor.
        dim: The dimension or dimensions along which to reduce.
            If this is not provided, all dimensions are reduced.
        keepdim: Whether to retain reduced dimensions in the output.
            If this is False, reduced dimensions will be squeezed.

    Returns:
        mean of the input tensor

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (2, 3))
        output = tp.mean(input, dim=1, keepdim=True)

        assert np.array_equal(output.numpy(), np.mean(np.arange(6, dtype=np.float32).reshape((2, 3)), axis=1, keepdims=True))
    """
    return mean_impl(input, dim, keepdim)


@export.public_api(document_under="tensor/reduction_operations")
def var(
    input: "tripy.Tensor", dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False, correction: int = 1
) -> "tripy.Tensor":
    r"""
    Returns a new tensor containing the variance of the elements of the input tensor along the specified dimension.

    The variance along a dimension is defined as:

    :math:`\sigma^2 = \Large \frac{1}{max(0, N - \text{correction})} \large \sum_{i=1}^N (x_i - \bar{x})^2`

    where :math:`N` is the length of the dimension, :math:`x_i` is the :math:`i^{th}` element along the dimension,
    and :math:`\bar{x}` is the mean.

    Args:
        input: The input tensor.
        dim: The dimension or dimensions along which to reduce.
            If this is not provided, all dimensions are reduced.
        keepdim: Whether to retain reduced dimensions in the output.
            If this is False, reduced dimensions will be squeezed.
        correction: Defaults to Bessel's correction.

    Returns:
        variance of the input tensor

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (2, 3))
        output = tp.var(input, dim=1, keepdim=True)

        torch_input = torch.arange(6, dtype=torch.float32).reshape((2, 3)) # doc: omit
        assert np.array_equal(output.numpy(), torch_input.var(dim=1, keepdim=True).numpy())
    """
    from tripy.frontend.trace.ops.binary_elementwise import maximum

    mean_val = mean(input, dim=dim, keepdim=dim is not None)
    sub = (input - mean_val) ** 2.0
    return mean_impl(sub, dim=dim, keepdim=keepdim, apply_to_divisor=lambda x: maximum(x - correction, 0))


def _arg_min_max_impl(tensor: "tripy.Tensor", kind: ArgMinMax.Kind, dim: int, keepdim: bool):
    from tripy.frontend.tensor import Tensor
    from tripy.frontend.trace.ops.unsqueeze import unsqueeze
    from tripy.frontend.trace.ops.iota import iota_like
    from tripy.frontend.trace.ops.reshape import reshape

    if dim is None:
        tensor = reshape(tensor, (-1,))
    indices = iota_like(tensor, dim if dim else 0, datatype.int32)
    out = Tensor.build([tensor, indices], ArgMinMax, dim, kind)
    if keepdim:
        if dim is None:
            # TODO(#96): Support dim=None, keepdim=True
            raise NotImplementedError("dim=None, keepdim=True is not supported yet.")
        out = unsqueeze(out, dim)
    return out


@export.public_api(document_under="tensor/reduction_operations")
def argmax(input: "tripy.Tensor", dim: Optional[int] = None, keepdim: bool = False) -> "tripy.Tensor":
    """
    Returns a new tensor containing the indices of maximum values of the input tensor along the specified dimension.
    If there are multiple maximum values, then the indices of the first maximum value are returned.

    Args:
        input: The input tensor.
        dim: The dimension along which to reduce.
            If this is not provided, the argmax indice of the flattened input is returned.
        keepdim: Whether to retain reduced dimensions in the output.
            If this is False, reduced dimensions will be squeezed.

    Returns:
        A new tensor of datatype of ``tp.int32``.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (2, 3))
        output = tp.argmax(input, 0)

        assert np.array_equal(output.numpy(), np.argmax(np.arange(6, dtype=np.float32).reshape((2, 3)), 0))
    """
    return _arg_min_max_impl(input, ArgMinMax.Kind.ARG_MAX, dim, keepdim)


@export.public_api(document_under="tensor/reduction_operations")
def argmin(input: "tripy.Tensor", dim: Optional[int] = None, keepdim: bool = False) -> "tripy.Tensor":
    """
    Returns a new tensor containing the indices of minimum values of the input tensor along the specified dimension.
    If there are multiple minimum values, then the indices of the first minimum value are returned.

    Args:
        input: The input tensor.
        dim: The dimension along which to reduce.
            If this is not provided, the argmin indice of the flattened input is returned.
        keepdim: Whether to retain reduced dimensions in the output.
            If this is False, reduced dimensions will be squeezed.

    Returns:
        A new tensor of datatype ``tp.int32``.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (2, 3))
        output = tp.argmin(input, 0)

        assert np.array_equal(output.numpy(), np.argmin(np.arange(6, dtype=np.float32).reshape((2, 3)), 0))
    """
    return _arg_min_max_impl(input, ArgMinMax.Kind.ARG_MIN, dim, keepdim)
