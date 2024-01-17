import math

from tripy.common import datatype
from tripy.common.exception import raise_error
from tripy.common.types import ShapeInfo
from tripy.frontend.ops import full, full_like, iota, iota_like, where


def ones(shape: ShapeInfo, dtype: datatype.dtype = datatype.float32):
    """
    Creates a Tensor with all elements set to 1.

    Args:
        shape: A list or tuple of integers
        dtype: Optional datatype of an element in the resulting Tensor.

    Returns:
        A Tensor with all elements set to 1.

    Example:
    ::

        a = tp.ones([2, 3])
        print(a)
        assert np.array_equal(a.numpy(), np.ones([2, 3], dtype=np.float32))
    """
    return full(shape, 1, dtype)


def zeros(shape: ShapeInfo, dtype: datatype.dtype = datatype.float32):
    """
    Creates a Tensor with all elements set to 0.

    Args:
        shape: A list or tuple of integers
        dtype: Optional datatype of an element in the resulting Tensor.

    Returns:
        A Tensor with all elements set to 0.

    Example:
    ::

        a = tp.zeros([2, 3])
        print(a)
        assert np.array_equal(a.numpy(), np.zeros([2, 3], dtype=np.float32))
    """
    return full(shape, 0, dtype)


def ones_like(input: "tripy.Tensor", dtype: datatype.dtype = None):
    """
    Creates a Tensor with all elements set to 1, with its shape (and dtype if not given) same as input.

    Args:
        input: input tensor
        dtype: Optional datatype of an element in the resulting Tensor.

    Returns:
        A Tensor with all elements set to 1.

    Example:
    ::

        t = tp.zeros([2, 3], dtype=tp.float32)
        a = tp.ones_like(t)
        print(a)
        assert np.array_equal(a.numpy(), np.ones([2, 3], dtype=np.float32))
    """
    return full_like(input, 1, dtype)


def zeros_like(input: "tripy.Tensor", dtype: datatype.dtype = None):
    """
    Creates a Tensor with all elements set to 0, with its shape (and dtype if not given) same as input.

    Args:
        input: input tensor
        dtype: Optional datatype of an element in the resulting Tensor.

    Returns:
        A Tensor with all elements set to 0.

    Example:
    ::

        t = tp.ones([2, 3], dtype=tp.float32)
        a = tp.zeros_like(t)
        print(a)
        assert np.array_equal(a.numpy(), np.zeros([2, 3], dtype=np.float32))
    """
    return full_like(input, 0, dtype)


def tril(input: "tripy.Tensor", diagonal: int = 0):
    """
    Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.

    Args:
        input: input tensor of shape (..., M, N)
        diagonal: diagonal above which to zero elements. diagonal = 0 is the main diagonal, a negative value is below it and a positive value is above.

    Returns:
        Lower triangle of input of same shape and dtype (..., M, N)

    Example:
    ::

        t = tp.Tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=tp.float32)
        out = tp.tril(t)
        print(out)
        assert np.array_equal(out.numpy(), np.tril(np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32)))
    """
    tri_mask = (iota_like(input, 0, datatype.int32) + full_like(input, diagonal, datatype.int32)) >= iota_like(
        input, 1, datatype.int32
    )
    zeros_tensor = zeros_like(input)
    return where(tri_mask, input, zeros_tensor)


def arange(start, stop=None, step=1, dtype: datatype.dtype = datatype.float32):
    """
    Creates a sequence of numbers that begins at `start` and extends by
    increments of `step` up to but not including `stop`.

    arange can be called with a varying number of positional arguments:
    `arange(stop)`: Values are generated within the half-open interval [0, stop)
    `arange(start, stop)`: Values are generated within the half-open interval [start, stop).
    `arange(start, stop, step)`: Values are generated within [start, stop), with spacing between values given by step.

    Args:
        start: start of the sequence, default is 0
        stop: end of the sequence, not included
        step: space between the values, default is 1
        dtype: dtype of the resulting Tensor

    Example:
    ::

        a = tp.arange(5)
        print(f"a: {a}")
        assert (a.numpy() == np.arange(5, dtype=np.float32)).all()

        b = tp.arange(0.5, 2.5)
        print(f"b: {b}")
        assert (b.numpy() == np.arange(0.5, 2.5, dtype=np.float32)).all()

        c = tp.arange(2.3, 0.8, -0.2)
        print(f"c: {c}")
        assert np.allclose(c.numpy(), np.arange(2.3, 0.8, -0.2, dtype=np.float32))
    """
    if stop is None:
        start, stop = 0, start
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
