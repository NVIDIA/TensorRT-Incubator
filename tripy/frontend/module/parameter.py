import tripy.frontend.utils as frontend_utils
from tripy.frontend.tensor import Tensor
from tripy.utils import export


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
