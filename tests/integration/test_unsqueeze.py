import cupy as cp
import numpy as np
import pytest

import tripy as tp


class TestUnsqueezeOp:
    @pytest.mark.parametrize("axis", [0, 2, 3])
    def test_unsqueeze_dynamic_op(self, axis):
        def func(a):
            return tp.unsqueeze(a, dim=axis)

        # TODO: DS blocked on https://gitlab-master.nvidia.com/initialdl/mlir-tensorrt/-/issues/635
        # compiler = tp.Compiler(func)
        # compiler.compile(tp.InputInfo(([2, 4, 6], 2, 2, 3), dtype=tp.float32))

        inp = np.ones((4, 2, 2, 3), dtype=np.float32)

        out = func(tp.Tensor(inp))
        assert tp.allclose(out, tp.Tensor(np.expand_dims(inp, axis=axis)))
