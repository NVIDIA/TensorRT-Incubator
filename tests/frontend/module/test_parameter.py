import numpy as np

import tripy as tp
from tripy.logging import logger
from tripy.common.device import device


class TestParameter:
    def test_is_marked_const(self, capsys):
        with logger.use_verbosity("flat_ir"):
            param = tp.Parameter(tp.Tensor([1, 2, 3], device=device("gpu")))

            @tp.jit
            def func(param):
                return param + param

            func(param)
            captured = capsys.readouterr()
            print(captured.out)
            assert "ConstantOp(data=[1, 2, 3])" in captured.out.strip()

    def test_is_instance_of_tensor(self):
        param = tp.Parameter(tp.Tensor([1, 2, 3]))
        assert isinstance(param, tp.Parameter)

        tensor = tp.Tensor([1, 2, 3])
        assert not isinstance(tensor, tp.Parameter)

    def test_is_equivalent_to_tensor(self):
        tensor = tp.Tensor([1, 2, 3])
        param = tp.Parameter(tensor)

        assert np.array_equal(param.numpy(), tensor.numpy())

    def test_can_construct_from_non_tensor(self):
        param = tp.Parameter([1, 2, 3])
        assert np.array_equal(param.numpy(), np.array([1, 2, 3], dtype=np.int32))
