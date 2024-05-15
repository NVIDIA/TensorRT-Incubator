from tripy.common.utils import get_element_type
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY

import numpy as np
import tripy.common.datatype


@TENSOR_METHOD_REGISTRY("numpy")
def numpy(self) -> np.ndarray:
    # TODO(#188): Replace Tensor.data() and np.array(...) with np.from_dlpack(...)
    data = self.data().data()
    dtype = np.int32 if get_element_type(data) == tripy.common.datatype.int32 else np.float32
    return np.array(data, dtype=dtype)
