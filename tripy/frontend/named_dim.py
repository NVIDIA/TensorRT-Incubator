from typing import Union, Tuple, Any, Optional, Sequence, Dict, Callable
from dataclasses import dataclass
from ..util import default


def validate_profile(method: Callable) -> Callable:
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        assert (
            self.is_valid()
        ), f"NamedDim '{self._name}' should have min<=opt<=max but got min={self.min}, opt={self.opt} and max={self.max} with runtime shape={self._runtime_value}"
        return result

    return wrapper


class NamedDim:
    """
    Represents a named dimension with the range of values it can take.
    The range of values allow the compiler to optimize the program with
    dynamic shapes while obeying the constraints of the dimension.

    Args:
        name : Dimension name
        runtime_value : Runtime shape of the dimension
        min: Optional[int]
        max: Optional[int]
        opt: Optional[int]
            min/max/opt values provide the dynamic range of this dimension which will be used by
            the compiler to optimize the program.
            If only one of these values are provided, the compiler will assume static shapes for this dimension.
            If only min and max are provided, the opt value will be inferred as the mid point between min and max.
    Example:
    ::
        batch = NamedDim("batch", 2)
        assert batch.min == batch.opt == batch.max == 2

        dim = NamedDim("dim", 3, min=2, opt=4, max=9))
        assert dim.min == 2
        assert dim.opt == 4
        assert dim.max == 9
    """

    def __init__(
        self,
        name: str,
        runtime_value: int,
        min: Optional[int] = None,
        opt: Optional[int] = None,
        max: Optional[int] = None,
    ):
        self._name = name
        self._runtime_value = runtime_value

        def get_opt_value():
            return int((self._max + self._min) / 2.0)

        self._min = default(min, self._runtime_value)
        self._max = default(max, self._runtime_value)
        self._opt = default(opt, get_opt_value())

        assert (
            self.is_valid()
        ), f"NamedDim '{self._name}' should have min<=opt<=max but got min={self.min}, opt={self.opt} and max={self.max} with runtime shape={self._runtime_value}"

    def is_valid(self):
        return (
            self._min <= self.opt
            and self.opt <= self.max
            and self._runtime_value >= self._min
            and self._runtime_value <= self._max
        )

    @property
    def min(self) -> int:
        return self._min

    @property
    def opt(self) -> int:
        return self._opt

    @property
    def max(self) -> int:
        return self._max

    @property
    def runtime_value(self) -> int:
        return self._runtime_value

    @runtime_value.setter
    @validate_profile
    def runtime_value(self, shape) -> None:
        """
        Set runtime shape for a NamedDim
        """
        self._runtime_value = shape

    def __repr__(self) -> str:
        return f'NamedDim(name="{self._name}", runtime_value={self._runtime_value}, min={self._min}, opt={self._opt}, max={self._max})'
