import numpy as np
import pytest

# Internal-only imports
from tripy.frontend import TensorExpression


class TestFunctional:
    @pytest.mark.skip(reason="The test requires mlir_tensorrt to be installed which is not done by default in tripy.")
    def test_add_two_tensors(self):
        arr = np.array([2, 3], dtype=np.float32)
        a = TensorExpression.tensor(arr)
        b = TensorExpression.tensor(np.ones(2, dtype=np.float32))

        c = a + b
        out = c + c
        out.__repr__()
