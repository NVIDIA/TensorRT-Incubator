import tripy as tp
from tripy.frontend.trace.ops import Plugin


class TestPlugin:
    def test_op(self):
        X = tp.iota((1, 2, 4, 4))
        rois = tp.Tensor([[0.0, 0.0, 9.0, 9.0], [0.0, 5.0, 4.0, 9.0]], dtype=tp.float32)
        batch_indices = tp.zeros((2,), dtype=tp.int32)

        out = tp.plugin(
            "ROIAlign_TRT", [X, rois, batch_indices], output_info=[(X.rank, X.dtype)], output_height=5, output_width=5
        )

        assert isinstance(out, tp.Tensor)
        assert isinstance(out.trace_tensor.producer, Plugin)
