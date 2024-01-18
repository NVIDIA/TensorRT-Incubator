import numpy as np
import pytest

import tripy as tp


class DummyOp(tp.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.param = tp.nn.Parameter(tensor)

    def __call__(self):
        return self.param


class Network(tp.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = tp.nn.Parameter(tp.ones(2, dtype=tp.float32))
        self.dummy1 = DummyOp(tp.zeros(2, dtype=tp.float32))
        self.dummy2 = DummyOp(tp.arange(2, dtype=tp.float32))

    def __call__(self):
        return self.param + self.dummy1() + self.dummy2()


class JitNetwork(tp.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = tp.nn.Parameter(tp.ones(2, dtype=tp.float32))
        self.dummy1 = DummyOp(tp.zeros(2, dtype=tp.float32))
        self.dummy2 = DummyOp(tp.arange(2, dtype=tp.float32))

    @tp.jit
    def __call__(self):
        return self.param + self.dummy1() + self.dummy2()


class JitArgsNetwork(tp.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = tp.nn.Parameter(tp.ones(2, dtype=tp.float32))
        self.dummy1 = DummyOp(tp.zeros(2, dtype=tp.float32))
        self.dummy2 = DummyOp(tp.arange(2, dtype=tp.float32))

    @tp.jit(dummy_arg=1)
    def __call__(self, tensor1, tensor2):
        return self.param + self.dummy1() + self.dummy2() + tensor1 + tensor2


@pytest.fixture(params=[(Network, ()), (JitNetwork, ()), (JitArgsNetwork, (1, 2))])
def network(request):
    call_args = request.param[1]
    inputs = [tp.Tensor(np.full(2, v, dtype=np.float32), device=tp.device("gpu")) for v in call_args]
    yield request.param[0](), call_args, inputs


class TestModule:
    def test_basic(self, network):
        test_net, call_args, inputs = network
        assert len(test_net._params.keys()) == 1
        assert len(test_net._modules.keys()) == 2

        result = np.array([1.0, 2.0]) + np.full(2, sum(call_args), dtype=np.float32)
        assert np.array_equal(test_net(*inputs).numpy(), result)
