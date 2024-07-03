import cupy as cp
import pytest

import tripy as tp


def compare_split_results(tp_out, reference_out):
    if isinstance(tp_out, list):
        assert isinstance(reference_out, tuple)
        assert len(tp_out) == len(reference_out)
        for i in range(len(tp_out)):
            assert cp.array_equal(cp.from_dlpack(tp_out[i]), cp.array(reference_out[i]))
    else:
        assert cp.array_equal(cp.from_dlpack(tp_out), cp.array(reference_out))


class TestSplitOp:

    @pytest.mark.parametrize(
        "use_jit",
        [False, True],
    )
    @pytest.mark.parametrize(
        "dims_a, split_params, reference_slices",
        [
            ((4,), (2, 0), lambda t: (t[:2], t[2:])),
            ((4,), (1, 0), lambda t: t[:]),
            ((4,), (4, 0), lambda t: (t[0:1], t[1:2], t[2:3], t[3:4])),
            # https://gitlab-master.nvidia.com/initialdl/mlir-tensorrt/-/issues/860
            # ((4,), ([1, 2], 0), lambda t: (t[:1], t[1:2], t[2:])),
            ((12, 12), (3, 1), lambda t: (t[:, :4], t[:, 4:8], t[:, 8:])),
            ((12, 12), ([3], 1), lambda t: (t[:, :3], t[:, 3:])),
            ((12, 12), (4, 0), lambda t: (t[:3, :], t[3:6, :], t[6:9, :], t[9:12, :])),
            ((3, 0), (5, 1), lambda t: (t[:, :0], t[:, 0:0], t[:, 0:0], t[:, 0:0], t[:, 0:0])),
        ],
    )
    def test_split_static(self, dims_a, split_params, reference_slices, use_jit):
        a_cp = cp.random.rand(*dims_a).astype(cp.float32)
        a = tp.Tensor(a_cp, device=tp.device("gpu"))

        def func(t):
            return tp.split(t, split_params[0], split_params[1])

        if use_jit:
            func = tp.jit(func)

        out = func(a)
        reference_out = reference_slices(a_cp)
        compare_split_results(out, reference_out)

    @pytest.mark.parametrize(
        "use_jit",
        [False, True],
    )
    @pytest.mark.parametrize(
        "dynamic_dims_a, split_params, reference_slices",
        [
            ((tp.dynamic_dim(4, min=2, max=6), tp.dynamic_dim(3, min=3, max=6)), (2, 0), lambda t: (t[:2], t[2:])),
        ],
    )
    @pytest.mark.skip(
        "This presently segfaults when the JIT is enabled, possibly due to"
        "underlying issues with slicing on dynamic dimensions"
    )
    def test_split_dynamic(self, dynamic_dims_a, split_params, reference_slices, use_jit):
        concrete_dims = tuple([d.runtime_value for d in dynamic_dims_a])
        a_cp = cp.random.rand(*concrete_dims).astype(cp.float32)
        a = tp.Tensor(a_cp, shape=dynamic_dims_a, device=tp.device("gpu"))

        def func(a):
            return tp.split(a, split_params[0], split_params[1])

        if use_jit:
            func = tp.jit(func)

        out = func(a)
        reference_out = reference_slices(a_cp)
        compare_split_results(out, reference_out)
