from tripy import utils

# We use the tensor method registry to define methods on the `Tensor` class out of line.
# This lets the method live alongside the trace operation and makes it a bit more modular
# to add new operations.
TENSOR_METHOD_REGISTRY = utils.FunctionRegistry()
