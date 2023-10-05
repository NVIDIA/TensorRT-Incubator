from typing import Union, Tuple, Any
from dataclasses import dataclass

ShapeTuple = Union[int, Tuple[int, int], Tuple[int, int, int]]


class NamedDim:
    """
    Represents a named dimension with the range of values it can take.
    The range of values allow the compiler to optimize the program with
    dynamic shapes while obeying the constraints of the dimension.

    Args:
        name : Dimension name
        dim_range :  Union[int, Tuple[int, int], Tuple[int, int, int]]
            Provide the dynamic range of this dimension which will be used by
            the compiler to optimize the program.
            If only integer is provided, the dimension will be considered as
            static shape.
            If tuple of integers are provided, the dimension will be assumed to
            vary along the dynamic range and the compiler will optimize the
            program specifically for dim_range[1] shape.

    Example:
    >>> print(NamedDim("batch", 2))
        NamedDim(name=batch, dim_range=2)
    >>> print(NamedDim("batch", (2,3)))
        NamedDim(name=batch, dim_range=(2, 3))
    >>> print(NamedDim("batch", (2,3,4)))
        NamedDim(name=batch, dim_range=(2, 3, 4))
    """

    def __init__(self, name: str, dim_range: ShapeTuple):
        self._name = name
        self._opt = self._min = self._max = -1
        if isinstance(dim_range, int):
            self._opt = self._min = self._max = dim_range
        elif isinstance(dim_range, Tuple):
            self._min = dim_range[0]
            if len(dim_range) == 2:
                self._opt = self._max = dim_range[1]
            elif len(dim_range) == 3:
                self._opt = dim_range[1]
                self._max = dim_range[2]
            else:
                raise ValueError("NamedDim can only accept tuple of 3 or less integers.")
        else:
            raise ValueError("NamedDim accepts only integer or tuple of 3 or less integers.")

    @property
    def min(self):
        return self._min

    @property
    def opt(self):
        return self._opt

    @property
    def max(self):
        return self._max

    def __repr__(self) -> str:
        return f'NamedDim(name="{self._name}", dim_range=({self._min}, {self._opt}, {self._max}))'
