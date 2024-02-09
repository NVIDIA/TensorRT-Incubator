import math
import numbers

from tripy import utils
from tripy.common import datatype
from tripy.common.exception import raise_error
from tripy.common.types import ShapeInfo
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.trace.ops.fill import full, full_like
from tripy.frontend.trace.ops.iota import iota, iota_like
from tripy.frontend.trace.ops.where import where


def ones(shape: ShapeInfo, dtype: datatype.dtype = datatype.float32) -> "tripy.Tensor":
    """
    Creates a Tensor of the specified shape and dtype with all elements set to 1.

    Args:
        shape: The desired shape of the tensor.
        dtype: Datatype of elements.

    Returns:
        A tensor of shape ``shape`` with all elements set to 1.

    .. code-block:: python
        :linenos:
        :caption: Example

        output = tp.ones([2, 3])

        assert np.array_equal(output.numpy(), np.ones([2, 3], dtype=np.float32))
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

    .. code-block:: python
        :linenos:
        :caption: Example

        output = tp.zeros([2, 3])

        assert np.array_equal(output.numpy(), np.zeros([2, 3], dtype=np.float32))
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

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.zeros([2, 3], dtype=tp.float32)
        output = tp.ones_like(input)

        assert np.array_equal(output.numpy(), np.ones([2, 3], dtype=np.float32))
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

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.ones([2, 3], dtype=tp.float32)
        output = tp.zeros_like(input)

        assert np.array_equal(output.numpy(), np.zeros([2, 3], dtype=np.float32))
    """
    return full_like(input, 0, dtype)


@TENSOR_METHOD_REGISTRY("tril")
def tril(self, diagonal: int = 0) -> "tripy.Tensor":
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

    .. code-block:: python
        :linenos:
        :caption: Main Diagonal

        input = tp.iota((5, 5)) + 1.
        output = input.tril()

        assert np.array_equal(output.numpy(), np.tril(input.numpy()))

    .. code-block:: python
        :linenos:
        :caption: Two Diagonals Above Main

        input = tp.iota((5, 5)) + 1. # doc: omit
        output = input.tril(diagonal=2)

        assert np.array_equal(output.numpy(), np.tril(input.numpy(), 2))

    .. code-block:: python
        :linenos:
        :caption: One Diagonal Below Main

        input = tp.iota((5, 5)) + 1. # doc: omit
        output = input.tril(diagonal=-1)

        assert np.array_equal(output.numpy(), np.tril(input.numpy(), -1))
    """
    tri_mask = (iota_like(self, 0, datatype.int32) + full_like(self, diagonal, datatype.int32)) >= iota_like(
        self, 1, datatype.int32
    )
    zeros_tensor = zeros_like(self)
    return where(tri_mask, self, zeros_tensor)


# Used for overloading
ARANGE_REGISTRY = utils.FunctionRegistry()


@ARANGE_REGISTRY("arange")
def arange(
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

    .. code-block:: python
        :linenos:
        :caption: Example

        output = tp.arange(0.5, 2.5)

        assert (output.numpy() == np.arange(0.5, 2.5, dtype=np.float32)).all()

    .. code-block:: python
        :linenos:
        :caption: Custom ``step`` Value

        output = tp.arange(2.3, 0.8, -0.2)

        assert np.allclose(output.numpy(), np.arange(2.3, 0.8, -0.2, dtype=np.float32))
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
def arange(stop: numbers.Number, dtype: "tripy.dtype" = datatype.float32) -> "tripy.Tensor":
    r"""
    Returns a 1D tensor containing a sequence of numbers in the half-open interval
    :math:`[0, \text{stop})` incrementing by 1.


    Args:
        stop: The exclusive upper bound of the values to generate.
        dtype: The desired datatype of the tensor.

    Returns:
        A tensor of shape :math:`[\text{stop}]`.

    .. code-block:: python
        :linenos:
        :caption: Example

        output = tp.arange(5)

        assert (output.numpy() == np.arange(5, dtype=np.float32)).all()
    """
    return arange(0, stop, dtype=dtype)


arange = ARANGE_REGISTRY["arange"]
