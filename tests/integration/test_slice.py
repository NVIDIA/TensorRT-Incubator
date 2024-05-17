import numpy as np
import pytest

import tripy as tp


class TestSliceOp:

    @pytest.mark.parametrize(
        "use_jit",
        [False, True],
    )
    @pytest.mark.parametrize(
        "dims_a, slice_func",
        [
            ((2,), lambda t: t[-1]),
            ((4,), lambda t: t[-2]),
            ((4,), lambda t: t[1:]),
            ((2, 3, 4), lambda t: t[1, 2, 3]),
            ((2, 3, 4), lambda t: t[:, :-1, 2]),
            ((1, 2, 1, 4), lambda t: t[:, 1, 0, 2:-1]),
            # ensure that if a slice upper bound is past the end, it is clamped
            ((2, 3, 4), lambda t: t[:3, :4, :5]),
            # TODO #156: implement when infer_rank is available on frontend tensor
            # The current way to dynamically add start,limit,stride content to slice params is very hacky and not worth adding right now.
            # ((2,3,4,5), lambda t: t[:1]),
            # TODO #162: Empty tensor test
            # ((2,3,4), lambda t: t[0:0,:-2,1:]),
        ],
    )
    def test_static_slice_op(self, dims_a, slice_func, use_jit):
        a_np = np.random.rand(*dims_a).astype(np.float32)
        a = tp.Tensor(a_np, device=tp.device("gpu"))

        def func(a):
            return slice_func(a)

        if use_jit:
            func = tp.jit(func)

        out = func(a)
        assert np.array_equal(out.numpy(), np.array(slice_func(a_np)))
