import cupy as cp

import tripy as tp
from tripy import utils


class TestPlugin:
    def test_gelu(self):
        inp = tp.iota((2, 2))
        out = tp.plugin(
            "CustomGeluPluginDynamic",
            [inp],
            output_info=[(inp.rank, inp.dtype)],
            # Plugin Parameters:
            type_id=0,
        )

        ref_out = tp.gelu(inp)

        assert cp.allclose(cp.from_dlpack(out), cp.from_dlpack(ref_out))

    def test_dynamic_shape_gelu(self):
        inp = tp.iota((2, 1, 4))
        # TODO: Remove this dynamic shapes hack:
        inp.eval()
        inp._dynamic_shape = utils.to_dims((2, tp.dynamic_dim(1, 1, 2, 3), 4))

        @tp.jit
        def gelu(X):
            return tp.plugin("CustomGeluPluginDynamic", [X], output_info=[(X.rank, X.dtype)], type_id=0)

        # TODO: Make sure there is no recompilation
        assert cp.allclose(cp.from_dlpack(gelu(inp)), cp.from_dlpack(tp.gelu(inp)))

        new_inp = tp.ones((2, 2, 4), dtype=tp.float32)
        assert cp.allclose(cp.from_dlpack(gelu(new_inp)), cp.from_dlpack(tp.gelu(new_inp)))
