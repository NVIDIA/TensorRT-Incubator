import numpy as np

import tripy as tp
from tripy.common.logging import LoggerModes
from tests import helper


class TestParameter:
    def test_is_marked_const(self, capsys):
        from tripy.common.logging import set_logger_mode, LoggerModes

        set_logger_mode(LoggerModes.IR)
        param = tp.nn.Parameter(tp.Tensor([1, 2, 3]))

        @tp.jit
        def func(param):
            return param + param

        func(param)
        captured = capsys.readouterr()
        assert "ConstantOp(data=[1 2 3], dtype=int32, device=gpu:0)" in captured.err.strip()

    def test_is_instance_of_tensor(self):
        param = tp.nn.Parameter(tp.Tensor([1, 2, 3]))
        assert isinstance(param, tp.nn.Parameter)

        tensor = tp.Tensor([1, 2, 3])
        assert not isinstance(tensor, tp.nn.Parameter)

    def test_is_equivalent_to_tensor(self):
        tensor = tp.Tensor([1, 2, 3])
        param = tp.nn.Parameter(tensor)

        assert np.array_equal(param.numpy(), tensor.numpy())

    def test_can_construct_from_non_tensor(self):
        param = tp.nn.Parameter([1, 2, 3])
        assert np.array_equal(param.numpy(), np.array([1, 2, 3], dtype=np.int32))
