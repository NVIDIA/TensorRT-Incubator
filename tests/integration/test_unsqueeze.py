import numpy as np
import pytest
import torch

from tripy.frontend import Tensor
import tripy as tp


class TestUnsqueezeOp:

    @pytest.mark.parametrize(
        "use_jit",
        [
            False,
            # TODO: DS blocked on https://gitlab-master.nvidia.com/initialdl/mlir-tensorrt/-/issues/635
            #    True
        ],
    )
    @pytest.mark.parametrize(
        "dims_a, axis",
        [
            ((tp.Dim(4, min=2, opt=4, max=6), 2), 0),
            ((tp.Dim(4, min=2, opt=4, max=6), 2, 2, 3), 2),
            ((tp.Dim(4, min=2, opt=4, max=6), 2, 2, 3), 3),
        ],
    )
    def test_unsqueeze_op(self, dims_a, axis, use_jit):
        def get_np_dims(dims, dim_func):
            return [dim_func(d) if isinstance(d, tp.Dim) else d for d in dims]

        a_np = np.random.rand(*get_np_dims(dims_a, lambda x: x.runtime_value)).astype(np.float32)
        a = tp.Tensor(a_np, shape=dims_a, device=tp.device("gpu"))

        def func(a):
            return a.unsqueeze(dim=axis)

        if use_jit:
            func = tp.jit(func)

        out = func(a)
        assert np.allclose(out.numpy(), np.array(np.expand_dims(a_np, axis=axis)))