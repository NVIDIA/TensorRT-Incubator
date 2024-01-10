from typing import List, Tuple, Union
from tripy.frontend.dim import Dim
from tripy.util import make_list, make_tuple


def to_dims(shape: Tuple[Union[int, Dim]]):
    """
    Convert the given shape tuple to a tuple of Dim objects.

    Args:
        shape (Tuple[Union[int, Dim]]): The input shape.

    Returns:
        Tuple[Dim]: The converted shape as a tuple of Dim objects.
    """
    if shape is None:
        return None

    return make_tuple(Dim(dim) if not isinstance(dim, Dim) else dim for dim in make_list(shape))
