from tripy.frontend.dim import Dim
from tripy.utils import make_list, make_tuple

from tripy.common.types import ShapeInfo


def to_dims(shape: ShapeInfo):
    """
    Convert the given shape tuple to a tuple of Dim objects.

    Args:
        shape (Tuple[Union[int, Dim]]): The input shape.

    Returns:
        Tuple[Dim]: The converted shape as a tuple of Dim objects.
    """
    if shape is None:
        return None

    dims = make_list(shape)
    for i in range(len(shape)):
        if not isinstance(shape[i], Dim):
            dims[i] = Dim(shape[i])
    return make_tuple(dims)
