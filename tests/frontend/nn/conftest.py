import numpy as np
import pytest

import tripy as tp


class DummyNestedOp(tp.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.param = tp.nn.Parameter(tensor)

    def __call__(self):
        return self.param


class DummyOp(tp.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.nested = DummyNestedOp(tensor)

    def __call__(self):
        return self.nested()


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
def all_network_modes(request):
    call_args = request.param[1]
    inputs = [tp.Tensor(np.full(2, v, dtype=np.float32), device=tp.device("gpu")) for v in call_args]
    yield request.param[0](), call_args, inputs


@pytest.fixture
def network():
    yield Network()
