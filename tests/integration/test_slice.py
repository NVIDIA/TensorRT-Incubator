import cupy as cp
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
            # flip one dimension
            ((2, 3, 4), lambda t: t[:, ::-1, :]),
            # negative step size that evenly and unevenly divides
            ((2, 3, 4), lambda t: t[:, :, ::-2]),
            ((2, 3, 4), lambda t: t[:, :, ::-3]),
            # both bounds given with negative step size
            ((10,), lambda t: t[8:2:-2]),
            # one bound given with negative step size
            ((10,), lambda t: t[8::-2]),
            ((10,), lambda t: t[:2:-2]),
            # both bounds with uneven step size
            ((10,), lambda t: t[8:2:-3]),
            # not the same thing as [10::-1] -- this one leaves off the last element
            ((10,), lambda t: t[10:0:-1]),
            # clamps the start index for negative step size
            ((10,), lambda t: t[1024:0:-1]),
            ((1, 2, 1, 4), lambda t: t[:, 1, 0, 2:-1]),
            # ensure that if a slice upper bound is past the end, it is clamped
            ((2, 3, 4), lambda t: t[:3, :4, :5]),
            # TODO #156: implement when infer_rank is available on frontend tensor
            # The current way to dynamically add start,limit,stride content to slice params is very hacky and not worth adding right now.
            # ((2,3,4,5), lambda t: t[:1]),
            # empty dimension
            ((2, 3, 4), lambda t: t[0:0, :-2, 1:]),
            # out of bounds slice
            ((2, 3, 4), lambda t: t[3:5, :-2, 1:]),
            # ensure that absurdly out of bounds dims will result in an empty tensor
            ((10,), lambda t: t[1234:5678]),
            # also check with negative step size
            ((10,), lambda t: t[1234:5678:-1]),
            # also in the usual index ordering
            ((10,), lambda t: t[5678:1234:-1]),
            # both out of bounds and negative (should be an empty tensor)
            ((5,), lambda t: t[-5:-12]),
            # also with negative step size
            ((5,), lambda t: t[-12:-5:-1]),
        ],
    )
    def test_static_slice_op(self, dims_a, slice_func, use_jit):
        a_cp = cp.random.rand(*dims_a).astype(np.float32)
        a = tp.Tensor(a_cp, device=tp.device("gpu"))

        def func(a):
            return slice_func(a)

        if use_jit:
            func = tp.jit(func)

        out = func(a)
        assert np.array_equal(cp.from_dlpack(out).get(), slice_func(a_cp).get())
