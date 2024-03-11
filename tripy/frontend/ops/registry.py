from typing import Any

from tripy import utils


# We use the tensor method registry to define methods on the `Tensor` class out of line.
# This lets the method live alongside the trace operation and makes it a bit more modular
# to add new operations. This can only be used for magic methods.
class TensorMethodRegistry(utils.FunctionRegistry):
    def __call__(self, key: Any):
        # We make a special exception for "shape" since we actually do want that to be a property
        assert (
            key == "shape" or key.startswith("__") and key.endswith("__")
        ), f"The tensor method registry should only be used for magic methods, but was used for: {key}"

        return super().__call__(key)


TENSOR_METHOD_REGISTRY = TensorMethodRegistry()
