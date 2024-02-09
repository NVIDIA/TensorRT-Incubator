# IMPORTANT: The `trace` submodule should *not* import the frontend `Tensor` as this
# will create a circular dependency.
from tripy.frontend.trace.trace import Trace
