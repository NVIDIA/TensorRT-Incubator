from typing import Any
from tripy import Tensor, jit
from tripy.frontend.nn import Module, Parameter

import numpy as np
import pytest
from tripy.common.logging import set_logger_mode, LoggerModes


class TestOp(Module):
    def __init__(self, np_array):
        super().__init__()
        self.param = Parameter(Tensor(np_array))

    def __call__(self):
        return self.param


class Network(Module):
    def __init__(self):
        super().__init__()
        self.param = Parameter(Tensor(np.ones(2, dtype=np.float32)))
        self.dummy1 = TestOp(np.zeros(2, dtype=np.float32))
        self.dummy2 = TestOp(np.arange(2, dtype=np.float32))

    def __call__(self):
        return self.param + self.dummy1() + self.dummy2()


@pytest.fixture
def testNet():
    return Network()


def test_parent_module(testNet):
    assert len(testNet._params.keys()) == 1
    assert len(testNet._modules.keys()) == 2

    assert (testNet().eval().cpu_view(np.float32) == np.array([1.0, 2.0])).all()


def test_nested_module_params(testNet):
    params = testNet.parameters()
    print(params)
    assert (params["param"].eval().cpu_view(np.float32) == np.array([1.0, 1.0], dtype=np.float32)).all()
    assert (params["dummy1.param"].eval().cpu_view(np.float32) == np.array([0.0, 0.0], dtype=np.float32)).all()
    assert (params["dummy2.param"].eval().cpu_view(np.float32) == np.array([0.0, 1.0], dtype=np.float32)).all()


def test_module_update_params(testNet):
    testNet.apply(lambda x: x + x)
    assert (testNet().eval().cpu_view(np.float32) == np.array([2.0, 4.0], dtype=np.float32)).all()


def test_module_save_load_params(testNet, tmp_path):
    # Use tmp_path fixture to create/delete temp file.
    file_path = tmp_path / f"weights.npz"
    testNet.save_weights(file_path)
    testNet.apply(lambda x: x + x + x)
    assert (testNet().eval().cpu_view(np.float32) == np.array([3.0, 6.0], dtype=np.float32)).all()

    testNet.load_weights(file_path)
    testNet.apply(lambda x: x + x)
    assert (testNet().eval().cpu_view(np.float32) == np.array([2.0, 4.0], dtype=np.float32)).all()
