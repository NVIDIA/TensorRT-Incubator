import tripy.frontend.utils as frontend_utils
from tripy.frontend.tensor import Tensor


class Parameter(Tensor):
    """
    A Parameter is a special kind of :class:`tripy.Tensor` that is treated by the compiler as a
    constant, enabling additional optimization opportunities.

    Example:

    .. code:: python
        :number-lines:

        param = tp.nn.Parameter(tp.Tensor([1.0, 1.0], dtype=tp.float32))
        print(param)
        assert isinstance(param, tp.nn.Parameter)
        assert isinstance(param, tp.Tensor)
    """

    @frontend_utils.convert_inputs_to_tensors()
    def __init__(self, tensor):
        self.__dict__ = tensor.__dict__
