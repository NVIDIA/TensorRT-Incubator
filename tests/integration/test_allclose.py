import torch
import pytest
import tripy as tp

from tests import helper


class TestAllClose:
    @pytest.mark.parametrize(
        "tensor_a, tensor_b, rtol, atol",
        [
            ([1e10, 1e-7], [1.00001e10, 1e-8], 1e-05, 1e-08),
            # TODO (#232): Reenable when fixed
            # ([1e10, 1e-8], [1.00001e10, 1e-9], 1e-05, 1e-08),
            ([1e10, 1e-8], [1.0001e10, 1e-9], 1e-05, 1e-08),
            ([1.0, 2.0, 3.0], [1.01, 2.01, 3.01], 0.0, 0.01),
            ([1.0, 2.0, 3.0], [1.01, 2.01, 3.01], 0.01, 0.0),
            ([1.0, 2.0, 3.0], [1.01, 2.01, 3.01], 0.01, 0.01),
        ],
    )
    def test_all_close_float32(self, tensor_a, tensor_b, rtol, atol):
        np_result = torch.allclose(torch.FloatTensor(tensor_a), torch.FloatTensor(tensor_b), rtol=rtol, atol=atol)
        tp_result = tp.allclose(
            tp.Tensor(tensor_a, dtype=tp.float32), tp.Tensor(tensor_b, dtype=tp.float32), rtol=rtol, atol=atol
        )
        assert np_result == tp_result
