import enum
from dataclasses import dataclass
from typing import Optional, Sequence, Union

from tripy import utils
from tripy.common import datatype
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy.utils import make_list


@dataclass(repr=False)
class Reduce(BaseTraceOp):
    """
    Represents a reduce operation.
    """

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

        from tripy.flat_ir.ops import ConstantOp, ReduceOp
        from tripy.flat_ir.tensor import FlatIRTensor
        from tripy.common.array import Array
        from tripy.common.device import device

        init_value = self.kind.init_value
        init_const = FlatIRTensor.build(shape=[], dtype=outputs[0].dtype, device=outputs[0].device)
        ConstantOp(
            self,
            [],
            [init_const],
            data=Array(
                np.array(init_value, dtype=outputs[0].dtype.name), outputs[0].dtype, shape=(), device=device("cpu")
            ),
        )
        ReduceOp(self, [inputs[0], init_const], outputs, reduce_mode=self.kind.op, reduce_dims=self.dim)


@dataclass(repr=False)
class ArgMinMax(Reduce):
    """
    Represents a argmin or argmax operation.
    """

    class Kind:
        ARG_MAX = "argmax"
        ARG_MIN = "argmin"

    dim: Sequence[int]
    kind: Kind

    def infer_dtypes(self):
        self.outputs[0].dtype = datatype.int32

    def to_flat_ir(self, inputs, outputs):
        import numpy as np

        from tripy.flat_ir.ops import ConstantOp, ArgMinMaxOp
        from tripy.flat_ir.tensor import FlatIRTensor
        from tripy.common.array import Array
        from tripy.common.device import device

        init_val_const = FlatIRTensor.build(shape=[], dtype=inputs[0].dtype, device=outputs[0].device)
        init_idx_const = FlatIRTensor.build(shape=[], dtype=outputs[0].dtype, device=outputs[0].device)

        ConstantOp(
            self,
            [],
            [init_val_const],
            data=Array(np.array(0, dtype=inputs[0].dtype.name), inputs[0].dtype, shape=(), device=device("cpu")),
        )
        ConstantOp(
            self,
            [],
            [init_idx_const],
            data=Array(np.array(0, dtype=outputs[0].dtype.name), outputs[0].dtype, shape=(), device=device("cpu")),
        )

        ArgMinMaxOp(
            self,
            [inputs[0], inputs[1], init_val_const, init_idx_const],
            outputs,
            reduce_mode=self.kind,
            reduce_dims=self.dim,
        )


def _reduce_impl(self, kind: Reduce.Kind, dim: Union[int, Sequence], keepdim: bool):
    from tripy.frontend import Tensor

    out = Tensor.build([self], Reduce, dim, kind)
    if keepdim:
        if dim is None:
            # TODO(#96): Support dim=None, keepdim=True
            raise NotImplementedError("dim=None, keepdim=True is not supported yet.")
        for d in sorted(make_list(dim)):
            out = out.unsqueeze(d)

    return out


def mean_impl(tensor: "tripy.Tensor", dim: Union[int, Sequence] = None, keepdim: bool = False, apply_to_divisor=None):
    sum = tensor.sum(dim=dim, keepdim=keepdim)
    # compute number of elements in the array and divide by number of elements in dims
    input_shape = tensor.shape
    nb_elements = input_shape.prod(dim=0, keepdim=True)
    nb_elements_in_mean_dim = 1

    if dim is not None:
        for d in make_list(dim):
            nb_elements_in_mean_dim = input_shape[d] * nb_elements_in_mean_dim
        divisor = nb_elements_in_mean_dim
    else:
        divisor = nb_elements

    if apply_to_divisor:
        divisor = apply_to_divisor(divisor)

    return sum / (divisor.to(datatype.float32))


@TENSOR_METHOD_REGISTRY("sum")
def sum(self, dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False) -> "tripy.Tensor":
    """
    Returns a new tensor containing the sum of the elements of this tensor along the specified dimension.

    Args:
        dim: The dimension or dimensions along which to reduce.
            If this is not provided, all dimensions are reduced.
        keepdim: Whether to retain reduced dimensions in the output.
            If this is False, reduced dimensions will be squeezed.

    Returns:
        A new tensor of the same data type as this tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(6, dtype=tp.float32).reshape((2, 3))
        output = input.sum(0)

        assert np.array_equal(output.numpy(), np.sum(np.arange(6, dtype=np.float32).reshape((2, 3)), 0))
    """
    return _reduce_impl(self, Reduce.Kind.SUM, dim, keepdim)


@TENSOR_METHOD_REGISTRY("max")
def max(self, dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False) -> "tripy.Tensor":
    """
    Returns a new tensor containing the maximum of the elements of this tensor along the specified dimension.

    Args:
        dim: The dimension or dimensions along which to reduce.
            If this is not provided, all dimensions are reduced.
        keepdim: Whether to retain reduced dimensions in the output.
            If this is False, reduced dimensions will be squeezed.

    Returns:
        A new tensor of the same data type as this tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(6, dtype=tp.float32).reshape((2, 3))
        output = input.max(0)

        assert np.array_equal(output.numpy(), np.max(np.arange(6, dtype=np.float32).reshape((2, 3)), 0))
    """
    return _reduce_impl(self, Reduce.Kind.MAX, dim, keepdim)


@TENSOR_METHOD_REGISTRY("prod")
def prod(self, dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False) -> "tripy.Tensor":
    """
    Returns a new tensor containing the product of the elements of this tensor along the specified dimension.

    Args:
        dim: The dimension or dimensions along which to reduce.
            If this is not provided, all dimensions are reduced.
        keepdim: Whether to retain reduced dimensions in the output.
            If this is False, reduced dimensions will be squeezed.

    Returns:
        A new tensor of the same data type as this tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(6, dtype=tp.float32).reshape((2, 3))
        output = input.prod(0)

        assert np.array_equal(output.numpy(), np.prod(np.arange(6, dtype=np.float32).reshape((2, 3)), 0))
    """
    return _reduce_impl(self, Reduce.Kind.MUL, dim, keepdim)


@TENSOR_METHOD_REGISTRY("mean")
def mean(self, dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False) -> "tripy.Tensor":
    """
    Returns a new tensor containing the mean of the elements of this tensor along the specified dimension.

    Args:
        dim: The dimension or dimensions along which to reduce.
            If this is not provided, all dimensions are reduced.
        keepdim: Whether to retain reduced dimensions in the output.
            If this is False, reduced dimensions will be squeezed.

    Returns:
        mean of the input tensor

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(6, dtype=tp.float32).reshape((2, 3))
        output = input.mean(dim=1, keepdim=True)

        assert np.array_equal(output.numpy(), np.mean(np.arange(6, dtype=np.float32).reshape((2, 3)), axis=1, keepdims=True))
    """
    return mean_impl(self, dim, keepdim)


@TENSOR_METHOD_REGISTRY("var")
def var(
    self, dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False, correction: int = 1
) -> "tripy.Tensor":
    r"""
    Returns a new tensor containing the variance of the elements of this tensor along the specified dimension.

    The variance along a dimension is defined as:

    :math:`\sigma^2 = \Large \frac{1}{max(0, N - \text{correction})} \large \sum_{i=1}^N (x_i - \bar{x})^2`

    where :math:`N` is the length of the dimension, :math:`x_i` is the :math:`i^{th}` element along the dimension,
    and :math:`\bar{x}` is the mean.

    Args:
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

        input = tp.arange(6, dtype=tp.float32).reshape((2, 3))
        output = input.var(dim=1, keepdim=True)

        torch_input = torch.arange(6, dtype=torch.float32).reshape((2, 3)) # doc: omit
        assert np.array_equal(output.numpy(), torch_input.var(dim=1, keepdim=True).numpy())
    """

    mean = self.mean(dim=dim, keepdim=dim is not None)
    sub = (self - mean) ** 2.0
    return mean_impl(sub, dim=dim, keepdim=keepdim, apply_to_divisor=lambda x: (x - correction).maximum(0))


def _arg_min_max_impl(self: "tripy.Tensor", kind: ArgMinMax.Kind, dim: int, keepdim: bool):
    from tripy.frontend import Tensor, iota_like

    # TODO(128): Use self.reshape((-1,)) to flatten input when dim=None
    if dim is None:
        raise NotImplementedError("dim=None is not supported yet.")
    indices = iota_like(self, dim if dim else 0, datatype.int32)
    out = Tensor.build([self, indices], ArgMinMax, dim, kind)
    if keepdim:
        if dim is None:
            # TODO(#96): Support dim=None, keepdim=True
            raise NotImplementedError("dim=None, keepdim=True is not supported yet.")
        out = out.unsqueeze(dim)
    return out


@TENSOR_METHOD_REGISTRY("argmax")
def argmax(self, dim: Optional[int] = None, keepdim: bool = False) -> "tripy.Tensor":
    """
    Returns a new tensor containing the indices of maximum values of this tensor along the specified dimension.
    If there are multiple maximum values, then the indices of the first maximum value are returned.

    Args:
        dim: The dimension along which to reduce.
            If this is not provided, the argmax indice of the flattened input is returned.
        keepdim: Whether to retain reduced dimensions in the output.
            If this is False, reduced dimensions will be squeezed.

    Returns:
        A new tensor of datatype of ``tp.int32``.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(6, dtype=tp.float32).reshape((2, 3))
        output = input.argmax(0)

        assert np.array_equal(output.numpy(), np.argmax(np.arange(6, dtype=np.float32).reshape((2, 3)), 0))
    """
    return _arg_min_max_impl(self, ArgMinMax.Kind.ARG_MAX, dim, keepdim)


@TENSOR_METHOD_REGISTRY("argmin")
def argmin(self, dim: Optional[int] = None, keepdim: bool = False) -> "tripy.Tensor":
    """
    Returns a new tensor containing the indices of minimum values of this tensor along the specified dimension.
    If there are multiple minimum values, then the indices of the first minimum value are returned.

    Args:
        dim: The dimension along which to reduce.
            If this is not provided, the argmin indice of the flattened input is returned.
        keepdim: Whether to retain reduced dimensions in the output.
            If this is False, reduced dimensions will be squeezed.

    Returns:
        A new tensor of datatype ``tp.int32``.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(6, dtype=tp.float32).reshape((2, 3))
        output = input.argmin(0)

        assert np.array_equal(output.numpy(), np.argmin(np.arange(6, dtype=np.float32).reshape((2, 3)), 0))
    """
    return _arg_min_max_impl(self, ArgMinMax.Kind.ARG_MIN, dim, keepdim)
