import pytest
import numpy as np

import tripy as tp
from tests import helper
from tripy.frontend.trace.ops import Reshape, Squeeze


class TestReshape:
    def test_op_func(self):
        a = tp.Tensor([1, 2, 3, 4])
        a = tp.reshape(a, (1, 1, 4))
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Reshape)

    def test_neg_dim_func(self):
        a = tp.Tensor([1, 2, 3, 4])
        a = tp.reshape(a, (1, 1, -1))
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Reshape)

    def test_infer_rank(self):
        a = tp.Tensor([1, 2, 3, 4])
        a = tp.reshape(a, (1, 1, -1))
        assert a.trace_tensor.rank == 3


class TestSqueeze:
    def test_op_func(self):
        a = tp.Tensor(np.ones((1, 1, 4), dtype=np.int32))
        a = tp.squeeze(a, dims=(0, 1))
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Squeeze)

    @pytest.mark.skip(
        "Program segfaulting instead of error being reported: https://gitlab-master.nvidia.com/initialdl/mlir-tensorrt/-/issues/855"
    )
    def test_incorrect_dims(self):
        a = tp.Tensor(np.ones((1, 1, 4), dtype=np.int32))
        b = tp.squeeze(a, 2)

        with helper.raises(
            tp.TripyException,
            match="Cannot select an axis to squeeze out which has size not equal to one",
            has_stack_info_for=[a, b],
        ):
            b.eval()

    def test_infer_rank(self):
        a = tp.ones((3, 2, 1, 2))
        b = tp.squeeze(a, 2)
        assert b.trace_tensor.rank == 3
