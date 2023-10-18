import inspect
import sys
import pytest
import numpy as np

from tripy import Tensor
from tripy import set_logger_mode, LoggerModes

# Internal-only imports
from tripy.frontend.parameters import BinaryElementwiseParameters, ValueParameters
from tripy.util.stack_info import SourceInfo


class TestFunctional:
    @pytest.mark.skip(reason="The test requires mlir_tensorrt to be installed which is not done by default in tripy.")
    def test_add_two_tensors(self):
        arr = np.array([2, 3], dtype=np.float32)
        a = Tensor.tensor(arr)
        b = Tensor.tensor(np.ones(2, dtype=np.float32))

        c = a + b
        out = c + c
        out.__repr__()
