from typing import Callable, Optional

from tripy.util import default


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
            runtime_value : Runtime shape of the dimension
            min:
            max:
            opt:
                min/max/opt values provide the dynamic range of this dimension which will be used by
                the compiler to optimize the program.
                If only one of these values are provided, the compiler will assume static shapes for this dimension.
                If only min and max are provided, the opt value will be inferred as the mid point between min and max.

        Example:
        ::
            from tripy.frontend import Dim

            batch = Dim(2)
            assert batch.min == batch.opt == batch.max == 2

            dim = Dim(3, min=2, opt=4, max=9)
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

    def is_static_shape(self):
        return self._min == self._max and self._min == self._opt and self._min == self._runtime_value

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
        if isinstance(other, Dim):
            return self.min == other.min and self.max == other.max and self.opt == other.opt
        return False

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
        return f"Dim(runtime_value={self._runtime_value}, min={self._min}, opt={self._opt}, max={self._max})"
