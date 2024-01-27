import math
import numbers

from tripy import utils
from tripy.common import datatype
from tripy.common.exception import raise_error
from tripy.common.types import ShapeInfo
from tripy.frontend.ops.fill import full, full_like
from tripy.frontend.ops.iota import iota, iota_like
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.ops.where import where


def ones(shape: ShapeInfo, dtype: datatype.dtype = datatype.float32) -> "tripy.Tensor":
    """
    Creates a Tensor of the specified shape and dtype with all elements set to 1.

    Args:
        shape: The desired shape of the tensor.
        dtype: Datatype of elements.

    Returns:
        A tensor of shape ``shape`` with all elements set to 1.

    Example:

    .. code:: python
        :number-lines:

        a = tp.ones([2, 3])
        print(a)
        assert np.array_equal(a.numpy(), np.ones([2, 3], dtype=np.float32))
    """
    return full(shape, 1, dtype)


def zeros(shape: ShapeInfo, dtype: datatype.dtype = datatype.float32) -> "tripy.Tensor":
    """
    Creates a Tensor of the specified shape and dtype with all elements set to 0.

    Args:
        shape: The desired shape of the tensor.
        dtype: Datatype of elements.

    Returns:
        A tensor of shape ``shape`` with all elements set to 0.

    Example:

    .. code:: python
        :number-lines:

        a = tp.zeros([2, 3])
        print(a)
        assert np.array_equal(a.numpy(), np.zeros([2, 3], dtype=np.float32))
    """
    return full(shape, 0, dtype)


def ones_like(input: "tripy.Tensor", dtype: datatype.dtype = None) -> "tripy.Tensor":
    """
    Creates a tensor with all elements set to 1 of the same shape as the input tensor.

    Args:
        input: The input tensor.
        dtype: Datatype of elements. If set to ``None``, the datatype of the input tensor is used.

    Returns:
        A tensor of the same shape as the input with all elements set to 1.

    Example:

    .. code:: python
        :number-lines:

        t = tp.zeros([2, 3], dtype=tp.float32)
        a = tp.ones_like(t)
        print(a)
        assert np.array_equal(a.numpy(), np.ones([2, 3], dtype=np.float32))
    """
    return full_like(input, 1, dtype)


def zeros_like(input: "tripy.Tensor", dtype: datatype.dtype = None) -> "tripy.Tensor":
    """
    Creates a Tensor with all elements set to 0 of the same shape as the input tensor.

    Args:
        input: The input tensor.
        dtype: Datatype of elements. If set to ``None``, the datatype of the input tensor is used.

    Returns:
        A tensor of the same shape as the input with all elements set to 0.

    Example:

    .. code:: python
        :number-lines:

        t = tp.ones([2, 3], dtype=tp.float32)
        a = tp.zeros_like(t)
        print(a)
        assert np.array_equal(a.numpy(), np.zeros([2, 3], dtype=np.float32))
    """
    return full_like(input, 0, dtype)


@TENSOR_METHOD_REGISTRY("tril")
def tril(self: "tripy.Tensor", diagonal: int = 0) -> "tripy.Tensor":
    r"""
    Returns the lower triangular part of each :math:`[M, N]` matrix in the tensor, with all other elements set to 0.
    If the tensor has more than two dimensions, it is treated as a batch of matrices.

    Args:
        diagonal: The diagonal above which to zero elements.
            ``diagonal=0`` indicates the main diagonal which is defined by the set of indices
            :math:`{{(i, i)}}` where :math:`i \in [0, min(M, N))`.

            Positive values indicate the diagonal which is that many diagonals above the main one,
            while negative values indicate one which is below.

    Returns:
        A tensor of the same shape and datatype as this tensor.

    For example, the lower triangular along the main diagonal:

    .. code:: python
        :number-lines:

        inp = tp.iota((5, 5)) + 1.
        print(f"inp: {inp}\n")

        out = inp.tril()
        print(f"out: {out}")

        assert np.array_equal(out.numpy(), np.tril(inp.numpy()))

    Along the diagonal that is two diagonals above the main:

    .. code:: python
        :number-lines:

        inp = tp.iota((5, 5)) + 1. # doc: omit
        out = inp.tril(diagonal=2)
        print(f"out: {out}")

        assert np.array_equal(out.numpy(), np.tril(inp.numpy(), 2))

    Along the diagonal that is one diagonal below the main:

    .. code:: python
        :number-lines:

        inp = tp.iota((5, 5)) + 1. # doc: omit
        out = inp.tril(diagonal=-1)
        print(f"out: {out}")

        assert np.array_equal(out.numpy(), np.tril(inp.numpy(), -1))
    """
    tri_mask = (iota_like(self, 0, datatype.int32) + full_like(self, diagonal, datatype.int32)) >= iota_like(
        self, 1, datatype.int32
    )
    zeros_tensor = zeros_like(self)
    return where(tri_mask, self, zeros_tensor)


# Used for overloading
ARANGE_REGISTRY = utils.FunctionRegistry()


@ARANGE_REGISTRY("arange")
def start_stop_step(
    start: numbers.Number, stop: numbers.Number, step: numbers.Number = 1, dtype: "tripy.dtype" = datatype.float32
) -> "tripy.Tensor":
    r"""
    Returns a 1D tensor containing a sequence of numbers in the half-open interval
    :math:`[0, \text{stop})` incrementing by :math:`\text{step}`.

    Args:
        start: The inclusive lower bound of the values to generate.
        stop: The exclusive upper bound of the values to generate.
        step: The spacing between values.
        dtype: The desired datatype of the tensor.

    Returns:
        A tensor of shape :math:`[\frac{\text{stop}-\text{start}}{\text{step}}]`.

    For example:

    .. code:: python
        :number-lines:

        out = tp.arange(0.5, 2.5)
        print(f"out: {out}")
        assert (out.numpy() == np.arange(0.5, 2.5, dtype=np.float32)).all()

    Using a different ``step`` value:

    .. code:: python
        :number-lines:


        out = tp.arange(2.3, 0.8, -0.2)
        print(f"out: {out}")
        assert np.allclose(out.numpy(), np.arange(2.3, 0.8, -0.2, dtype=np.float32))
    """
    if step == 0:
        raise_error("Step in arange cannot be 0.", [])

    size = math.ceil((stop - start) / step)
    if size <= 0:
        raise_error(
            "Arange tensor is empty.",
            details=[
                f"start={start}, stop={stop}, step={step}",
            ],
        )
    output = iota((size,), 0, dtype) * full((size,), step, dtype) + full((size,), start, dtype)
    return output


@ARANGE_REGISTRY("arange")
def stop_only(stop: numbers.Number, dtype: "tripy.dtype" = datatype.float32) -> "tripy.Tensor":
    r"""
    Returns a 1D tensor containing a sequence of numbers in the half-open interval
    :math:`[0, \text{stop})` incrementing by 1.


    Args:
        stop: The exclusive upper bound of the values to generate.
        dtype: The desired datatype of the tensor.

    Returns:
        A tensor of shape :math:`[\text{stop}]`.

    For example:

    .. code:: python
        :number-lines:

        out = tp.arange(5)
        print(f"out: {out}")
        assert (out.numpy() == np.arange(5, dtype=np.float32)).all()
    """
    return arange(0, stop, dtype=dtype)


arange = ARANGE_REGISTRY["arange"]
