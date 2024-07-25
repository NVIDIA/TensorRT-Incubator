from typing import Sequence

import tripy.frontend.utils as frontend_utils
from tripy import export, utils
from tripy.frontend.tensor import Tensor
from tripy.utils import Result


@export.public_api(document_under="modules", autodoc_options=[":no-members:", ":no-special-members:"])
class Parameter(Tensor):
    """
    A Parameter is a special kind of :class:`tripy.Tensor` that is treated by the compiler as a
    constant, enabling additional optimization opportunities.
    """

    @frontend_utils.convert_inputs_to_tensors()
    def __init__(self, tensor: "tripy.Tensor") -> None:
        """
        Args:
            tensor: The tensor value for this parameter.

        .. code-block:: python
            :linenos:
            :caption: Example

            parameter = tp.Parameter(tp.Tensor([1.0, 1.0], dtype=tp.float32))

            assert isinstance(parameter, tp.Parameter)
            assert isinstance(parameter, tp.Tensor)
        """
        self.__dict__ = tensor.__dict__

    def _is_compatible_helper(self, original_shape, other_shape, original_dtype, other_dtype) -> Result:
        if original_shape != other_shape:
            return Result.err(
                ["New parameter shape: ", other_shape, " is not compatible with current shape: ", original_shape]
            )
        if original_dtype != other_dtype:
            return Result.err(
                ["New parameter dtype: ", other_dtype, " is not compatible with current dtype: ", original_dtype]
            )
        return Result.ok()

    def _is_compatible(self, other: "Parameter") -> Result:
        # Determines whether another parameter has the same shape and
        # data type as this one.
        return self._is_compatible_helper(self.shape.eval(), other.shape.eval(), self.dtype, other.dtype)


class DefaultParameter(Parameter):
    """
    Behaves exactly like a parameter except does not cause
    data to be materialized for shape/dtype checks.
    Useful for initializing module parameters.
    """

    def __init__(self, shape: Sequence[int], dtype: "tripy.dtype") -> None:
        from tripy.frontend.ops.tensor_initializers import arange
        from tripy.frontend.trace.ops.reshape import reshape

        super().__init__(reshape(arange(utils.volume(shape), dtype), shape))

        self._shape = shape
        self._dtype = dtype

    def _is_compatible(self, other: "Parameter") -> Result:
        return self._is_compatible_helper(
            tuple(self._shape), tuple(other.shape.eval().data()), self._dtype, other.dtype
        )
