from tripy.common import datatype
from tripy.common.types import ShapeInfo
from tripy.frontend.ops.fill import full, full_like
from tripy.frontend.ops.iota import arange_like
from tripy.frontend.ops.where import where


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

        import numpy as np

        a = tp.ones([2, 3])
        assert (a.numpy() == np.ones([2, 3], dtype=np.float32)).all()
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

        import numpy as np

        a = tp.zeros([2, 3])
        assert (a.numpy() == np.zeros([2, 3], dtype=np.float32)).all()
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

        import numpy as np

        t = tp.Tensor(np.zeros([2, 3], dtype=np.float32))
        a = tp.ones_like(t)
        assert (a.numpy() == np.ones([2, 3], dtype=np.float32)).all()
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

        import numpy as np

        t = tp.Tensor(np.ones([2, 3], dtype=np.float32))
        a = tp.zeros_like(t)
        assert (a.numpy() == np.zeros([2, 3], dtype=np.float32)).all()
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

        import numpy as np

        t = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32)
        a = tp.tril(tp.Tensor(t))
        assert (a.numpy() == np.tril(t)).all()
    """
    tri_mask = (arange_like(input, 0, datatype.int32) + full_like(input, diagonal, datatype.int32)) >= arange_like(
        input, 1, datatype.int32
    )
    zeros_tensor = zeros_like(input)
    return where(tri_mask, input, zeros_tensor)
