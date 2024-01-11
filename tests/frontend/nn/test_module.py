import numpy as np
import pytest

from tripy import Tensor
from tripy import device, jit
from tripy.frontend.nn import Module, Parameter


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


class JitNetwork(Module):
    def __init__(self):
        super().__init__()
        self.param = Parameter(Tensor(np.ones(2, dtype=np.float32)))
        self.dummy1 = TestOp(np.zeros(2, dtype=np.float32))
        self.dummy2 = TestOp(np.arange(2, dtype=np.float32))

    @jit
    def __call__(self):
        return self.param + self.dummy1() + self.dummy2()


class JitArgsNetwork(Module):
    def __init__(self):
        super().__init__()
        self.param = Parameter(Tensor(np.ones(2, dtype=np.float32)))
        self.dummy1 = TestOp(np.zeros(2, dtype=np.float32))
        self.dummy2 = TestOp(np.arange(2, dtype=np.float32))

    @jit(dummy_arg=1)
    def __call__(self, tensor1, tensor2):
        return self.param + self.dummy1() + self.dummy2() + tensor1 + tensor2


def _create_test_tensor(call_args):
    if not call_args:
        return call_args
    return [Tensor(np.full(2, v, dtype=np.float32), device=device("gpu")) for v in call_args]


@pytest.mark.parametrize("test_net, call_args", [(Network(), ()), (JitNetwork(), ()), (JitArgsNetwork(), (1, 2))])
def test_parent_module(test_net, call_args):
    assert len(test_net._params.keys()) == 1
    assert len(test_net._modules.keys()) == 2

    result = np.array([1.0, 2.0]) + np.full(2, sum(call_args), dtype=np.float32)
    assert (test_net(*_create_test_tensor(call_args)).numpy() == result).all()


@pytest.mark.parametrize("test_net", [Network(), JitNetwork(), JitArgsNetwork()])
def test_nested_module_params(test_net):
    params = test_net.parameters()
    print(params)
    assert (params["param"].numpy() == np.array([1.0, 1.0], dtype=np.float32)).all()
    assert (params["dummy1.param"].numpy() == np.array([0.0, 0.0], dtype=np.float32)).all()
    assert (params["dummy2.param"].numpy() == np.array([0.0, 1.0], dtype=np.float32)).all()


@pytest.mark.parametrize("test_net, call_args", [(Network(), ()), (JitNetwork(), ()), (JitArgsNetwork(), (1, 2))])
def test_module_update_params(test_net, call_args):
    f = lambda x: x + x
    test_net.apply(f)
    result = f(np.array([1.0, 2.0])) + np.full(2, sum(call_args), dtype=np.float32)
    assert (test_net(*_create_test_tensor(call_args)).numpy() == result).all()


@pytest.mark.parametrize("test_net, call_args", [(Network(), ()), (JitNetwork(), ()), (JitArgsNetwork(), (1, 2))])
def test_module_save_load_params(test_net, call_args, tmp_path):
    # Use tmp_path fixture to create/delete temp file.
    file_path = tmp_path / f"weights.npz"
    test_net.save_weights(file_path)
    f1 = lambda x: x + x + x
    test_net.apply(f1)
    result = f1(np.array([1.0, 2.0])) + np.full(2, sum(call_args), dtype=np.float32)
    assert (test_net(*_create_test_tensor(call_args)).numpy() == result).all()

    test_net.load_weights(file_path)
    f2 = lambda x: x + x
    test_net.apply(f2)
    result = f2(np.array([1.0, 2.0])) + np.full(2, sum(call_args), dtype=np.float32)
    assert (test_net(*_create_test_tensor(call_args)).numpy() == result).all()
