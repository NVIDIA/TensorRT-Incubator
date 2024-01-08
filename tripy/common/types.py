from collections import namedtuple
from typing import Sequence

ShapeInfo = Sequence[int]
TensorInfo = namedtuple("TensorInfo", ["shape", "dtype"])
