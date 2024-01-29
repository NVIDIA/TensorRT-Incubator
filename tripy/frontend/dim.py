import functools
from typing import Callable, Optional

from tripy.common.exception import raise_error
from tripy.utils import default


def validate_profile(method: Callable) -> Callable:
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)

        if not self.is_valid():
            raise_error(
                "Invalid arguments to `Dim()`.",
                details=[
                    f"Dim should have `min`<={{`opt`,`runtime_value`}}<=`max` but got min={self.min}, opt={self.opt}, max={self.max} "
                    f"with runtime_value={self._runtime_value}."
                ],
            )
        return result

    return wrapper


class Dim:
    """
    Represents the size of a dimension along with the range of values it can take.

    Specifying a range of values allows the compiler to optimize the program such
    that is is valid for any dimension size within that range.
    """

    @validate_profile
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
                    This should generally be the value you expect will be the most frequently occurring.
            max: Maximum value of the dimension.

        ``min``/``opt``/``max`` indicate the dynamic range of the dimension and are used by the compiler
        when optimizing the program.

        If only ``min`` and ``max`` are provided, the ``opt`` value will be inferred as
        the midpoint between the two.

        Creating a static dimension:

        .. code:: python

            batch = tp.Dim(2)

            assert batch.min == batch.opt == batch.max == 2

        Creating a dynamic dimension:

        .. code:: python

            dyn_batch = tp.Dim(3, min=2, opt=4, max=9)

            assert dyn_batch.min == 2
            assert dyn_batch.opt == 4
            assert dyn_batch.max == 9
        """
        self._runtime_value = runtime_value

        self._min = default(min, self._runtime_value)
        self._max = default(max, self._runtime_value)

        self._opt = default(opt, int((self._max + self._min) / 2.0))

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
        return f"{self.runtime_value}"

    def is_a_subset_of(self, cached) -> bool:
        if self.is_dynamic_dim():
            return self.min > cached.min and self.max < cached.max
        return self.runtime_value >= cached.min and self.runtime_value <= cached.max
