from typing import Callable, Optional

from tripy.utils import default


def validate_profile(method: Callable) -> Callable:
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        assert (
            self.is_valid()
        ), f"Dim should have min<=opt<=max but got min={self.min}, opt={self.opt} and max={self.max} with runtime shape={self._runtime_value}"
        return result

    return wrapper


class Dim:
    """
    Represents a dimension with the range of values it can take.
    The range of values allow the compiler to optimize the program with
    dynamic shapes while obeying the constraints of the dimension.
    """

    def __init__(
        self,
        runtime_value: int,
        min: Optional[int] = None,
        opt: Optional[int] = None,
        max: Optional[int] = None,
    ):
        """
        Args:
            runtime_value : Runtime value of the dimension.
            min: Minimum value of the dimension.
            opt: Value of the dimension for which to optimize.
            max: Maximum value of the dimension.

        ``min``/``max``/``opt`` values provide the dynamic range of this dimension.
        These will be used by the compiler to optimize the program.
        If only one of these values is provided, it must be the same as `runtime_value` and the compiler
        will assume static shapes for this dimension.
        If only min and max are provided, the opt value will be inferred as the mid point between min and max.

        Example:
        ::

            batch = tp.Dim(2)
            assert batch.min == batch.opt == batch.max == 2

            dim = tp.Dim(3, min=2, opt=4, max=9)
            assert dim.min == 2
            assert dim.opt == 4
            assert dim.max == 9
        """
        self._runtime_value = runtime_value

        def get_opt_value():
            return int((self._max + self._min) / 2.0)

        self._min = default(min, self._runtime_value)
        self._max = default(max, self._runtime_value)
        self._opt = default(opt, get_opt_value())

        assert (
            self.is_valid()
        ), f"Dim should have min<=opt<=max but got min={self.min}, opt={self.opt} and max={self.max} with runtime shape={self._runtime_value}"

    def is_valid(self):
        return (
            self._min <= self._opt
            and self._opt <= self._max
            and self._runtime_value >= self._min
            and self._runtime_value <= self._max
        )

    def is_static_dim(self):
        return self._runtime_value == self._min == self._opt == self._max

    def is_dynamic_dim(self):
        return self._runtime_value == -1 or not self.is_static_dim()

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

    def __eq__(self, other):
        if not isinstance(other, Dim):
            return not self.is_dynamic_dim() and self.min == other
        return self.min == other.min and self.max == other.max and self.opt == other.opt

    def __gt__(self, other):
        assert self.is_static_dim() and other.is_static_dim()
        return self.min > other.min

    def __hash__(self):
        return hash((self.min, self.max, self.opt, self.runtime_value))

    @runtime_value.setter
    @validate_profile
    def runtime_value(self, shape) -> None:
        """
        Set runtime shape for a Dim
        """
        self._runtime_value = shape

    def __repr__(self) -> str:
        if self.is_dynamic_dim():
            return f"Dim(runtime_value={self._runtime_value}, min={self._min}, opt={self._opt}, max={self._max})"
        else:
            return f"{self.runtime_value}"
