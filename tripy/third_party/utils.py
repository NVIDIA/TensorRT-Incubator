from tripy.common.utils import get_element_type
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY

import numpy as np
import tripy.common.datatype


@TENSOR_METHOD_REGISTRY("numpy")
def numpy(self) -> np.ndarray:
    import numpy as np
    import cupy as cp

    self.eval()
    assert self.device is not None

    try:
        return cp.from_dlpack(self).get() if self.device == tripy.common.device("gpu") else np.from_dlpack(self)
    except Exception as e:
        raise e
