import numpy as np
import pytest

import tripy as tp


def create_random_matrix(shape):
    return np.random.rand(*shape).astype(np.float32)


class TestLinear:
    @pytest.mark.parametrize(
        "use_jit",
        [False, True],
    )
    def test_linear_module(self, use_jit):
        class Network(tp.Module):
            def __init__(self):
                super().__init__()
                self.linear = tp.Linear(4, 2)

            def __call__(self, x):
                return self.linear(x)

        net = Network()
        a1 = tp.Tensor(np.ones((3, 4), dtype=np.float32), device=tp.device("gpu"))

        if use_jit:
            net = tp.jit(net)

        out = net(a1)

        np_out = np.ones((3, 4), dtype=np.float32) @ (np.ones((2, 4), dtype=np.float32).transpose()) + np.ones(
            (1, 2), dtype=np.float32
        )

        assert (out.numpy() == np.array(np_out)).all()
