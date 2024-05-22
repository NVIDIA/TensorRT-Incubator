import numpy as np
import cupy as cp
import pytest
import tripy as tp

from tests import helper


class TestConcatenate:
    @pytest.mark.parametrize(
        "tensor_shapes, dim, expected_shape",
        [
            ([(2, 3, 4), (2, 4, 4)], 1, (2, 7, 4)),
            ([(2, 3, 4), (2, 3, 2)], -1, (2, 3, 6)),
            ([(2, 3, 4), (2, 3, 4)], 0, (4, 3, 4)),
            ([(2, 3, 4)], 0, (2, 3, 4)),
        ],
    )
    def test_static_concat(self, tensor_shapes, dim, expected_shape):
        tensors = [tp.ones(shape) for shape in tensor_shapes]
        out = tp.concatenate(tensors, dim=dim)
        assert np.array_equal(
            cp.from_dlpack(out).get(), np.concatenate([np.ones(shape) for shape in tensor_shapes], axis=dim)
        )

    @pytest.mark.parametrize(
        "tensor_shapes, dim",
        [([(2, 3, 4), (2, 4, 4)], 0), ([(4, 5, 6), (4, 1, 6)], -1)],
    )
    def test_negative_concat(self, tensor_shapes, dim):
        tensors = [tp.ones(shape) for shape in tensor_shapes]
        with helper.raises(tp.TripyException, match=f"not compatible at non-concat index"):
            out = tp.concatenate(tensors, dim=dim)
            print(out)

    @pytest.mark.skip(reason="Todo: #156 Enable this test once infer_shapes is removed.")
    @pytest.mark.parametrize(
        "dims_a, dims_b, dim",
        [
            ((tp.dynamic_dim(4, min=2, opt=4, max=6), 2), (tp.dynamic_dim(4, min=2, opt=4, max=6), 2), 0),
            ((tp.dynamic_dim(4, min=2, opt=4, max=6), 2), (tp.dynamic_dim(4, min=2, opt=4, max=6), 2), 1),
            ((tp.dynamic_dim(4, min=2, opt=4, max=6), 2), (3, 2), 1),
        ],
    )
    def test_dynamic_concat(self, dims_a, dims_b, dim):
        def get_np_dims(dims, dim_func):
            return [dim_func(d) if isinstance(d, tp.dynamic_dim) else d for d in dims]

        a_cp = cp.random.rand(*get_np_dims(dims_a, lambda x: x.runtime_value)).astype(cp.float32)
        b_cp = cp.random.rand(*get_np_dims(dims_b, lambda x: x.runtime_value)).astype(cp.float32)

        a = tp.Tensor(a_cp, shape=dims_a, device=tp.device("gpu"))
        b = tp.Tensor(b_cp, shape=dims_b, device=tp.device("gpu"))

        @tp.jit
        def func(a, b):
            return tp.concatenate([a, b], dim=dim)

        out = func(a, b)
        print(out)
