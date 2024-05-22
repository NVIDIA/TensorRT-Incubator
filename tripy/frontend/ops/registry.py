from typing import Any

from tripy.function_registry import FunctionRegistry


# We use the tensor method registry to define methods on the `Tensor` class out of line.
# This lets the method live alongside the trace operation and makes it a bit more modular
# to add new operations. This can only be used for magic methods.
class TensorMethodRegistry(FunctionRegistry):
    def __call__(self, key: Any):
        # We make a special exception for "shape" since we actually do want that to be a property
        allowed_methods = ["numpy", "cupy", "shape"]
        assert (
            key in allowed_methods or key.startswith("__") and key.endswith("__")
        ), f"The tensor method registry should only be used for magic methods, but was used for: {key}"

        return super().__call__(key)


TENSOR_METHOD_REGISTRY = TensorMethodRegistry()
