from collections import namedtuple
from typing import Sequence

ShapeInfo = Sequence[int]
TypeInfo = namedtuple("TypeInfo", ["shape", "dtype"])
