from collections import namedtuple
from typing import Tuple, Union

from tripy.frontend.dim import Dim

ShapeInfo = Tuple[Union[int, Dim]]
TensorInfo = namedtuple("TensorInfo", ["shape", "dtype"])
