import functools
from typing import Any, Callable, Dict, Optional

from tripy import export, utils
from tripy.common.exception import raise_error
from tripy.utils.json import Decoder, Encoder


def validate_profile(method: Callable) -> Callable:
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)

        if not self.is_valid():
            raise_error(
                "Invalid arguments to `dynamic_dim()`.",
                details=[
                    f"dynamic_dim should have `min`<={{`opt`,`runtime_value`}}<=`max` but got min={self.min}, opt={self.opt}, max={self.max} "
                    f"with runtime_value={self._runtime_value}."
                ],
            )
        return result

    return wrapper


@export.public_api()
class dynamic_dim:
    """
    Represents the size of a dynamic dimension along with the range of values it can take.

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
    ) -> None:
        """
        Args:
            runtime_value : Runtime value of the dimension.
            min: Minimum value of the dimension.
            opt: Value of the dimension for which to optimize.
                    This should generally be the value you expect will be the most frequently occurring.
            max: Maximum value of the dimension.

        If only ``min`` and ``max`` are provided, the ``opt`` value will be inferred as
        the midpoint between the two. If only the ``runtime_value`` is provided, ``min``,
        ``opt``, and ``max`` will all be set to ``runtime_value``.

        .. code-block:: python
            :linenos:
            :caption: Example

            dyn_batch = tp.dynamic_dim(3, min=2, opt=4, max=9)

            assert dyn_batch.min == 2
            assert dyn_batch.opt == 4
            assert dyn_batch.max == 9
        """
        self._runtime_value = runtime_value

        self._min = utils.default(min, self._runtime_value)
        self._max = utils.default(max, self._runtime_value)
        self._opt = utils.default(opt, int((self._max + self._min) / 2.0))

    def is_valid(self):
        return (
            self._min <= self._opt
            and self._opt <= self._max
            and self._runtime_value >= self._min
            and self._runtime_value <= self._max
        )

    def is_dynamic_dim(self):
        all_values_same = self._runtime_value == self._min == self._opt == self._max
        return self._runtime_value == -1 or not all_values_same

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
        if not isinstance(other, dynamic_dim):
            return not self.is_dynamic_dim() and self.min == other
        return self.min == other.min and self.max == other.max and self.opt == other.opt

    def __gt__(self, other):
        if not isinstance(other, dynamic_dim):
            return self.min > other
        else:
            return self.min > other.max

    @runtime_value.setter
    @validate_profile
    def runtime_value(self, shape) -> None:
        """
        Set runtime shape for a dynamic_dim
        """
        self._runtime_value = shape

    def __repr__(self) -> str:
        if self.is_dynamic_dim():
            return (
                f"dynamic_dim(runtime_value={self._runtime_value}, min={self._min}, opt={self._opt}, max={self._max})"
            )
        return f"{self.runtime_value}"


@Encoder.register(dynamic_dim)
def encode(dim: dynamic_dim) -> Dict[str, Any]:
    return {"runtime_value": dim.runtime_value, "min": dim.min, "opt": dim.opt, "max": dim.max}


@Decoder.register(dynamic_dim)
def decode(dct: Dict[str, Any]) -> dynamic_dim:
    return dynamic_dim(dct["runtime_value"], min=dct["min"], opt=dct["opt"], max=dct["max"])
