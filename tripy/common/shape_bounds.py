from dataclasses import dataclass
from typing import Sequence


@dataclass
class ShapeBounds:
    min: Sequence[int]
    opt: Sequence[int]
    max: Sequence[int]
