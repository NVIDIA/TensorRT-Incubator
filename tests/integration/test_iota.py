import cupy as cp
import numpy as np
import pytest

import tripy as tp
from tests import helper


class TestIota:
    def _compute_ref_iota(self, dtype, shape, dim):
        if dim is None:
            dim = 0
        elif dim < 0:
            dim += len(shape)
        expected = np.arange(0, shape[dim], dtype=dtype)
        if dim < len(shape) - 1:
            expand_dims = [1 + i for i in range(len(shape) - 1 - dim)]
            expected = np.expand_dims(expected, expand_dims)
        expected = np.broadcast_to(expected, shape)
        return expected

    @pytest.mark.parametrize("dtype", [tp.float32, tp.int32, tp.float16, tp.int8])
    @pytest.mark.parametrize(
        "shape, dim",
        [
            ((2, 3), 1),
            ((2, 3), None),
            ((2, 3), -1),
            ((2, 3, 4), 2),
        ],
    )
    def test_iota(self, dtype, shape, dim):
        if dim:
            output = tp.iota(shape, dim, dtype)
        else:
            output = tp.iota(shape, dtype=dtype)

        assert np.array_equal(cp.from_dlpack(output).get(), self._compute_ref_iota(dtype.name, shape, dim))

    @pytest.mark.parametrize("dtype", [tp.float32, tp.int32, tp.float16, tp.int8])
    @pytest.mark.parametrize(
        "shape, dim",
        [
            ((2, 3), 1),
            ((2, 3), None),
            ((2, 3), -1),
            ((2, 3, 4), 2),
        ],
    )
    def test_iota_like(self, dtype, shape, dim):
        if dim:
            output = tp.iota_like(tp.ones(shape), dim, dtype)
        else:
            output = tp.iota_like(tp.ones(shape), dtype=dtype)

        assert np.array_equal(cp.from_dlpack(output).get(), self._compute_ref_iota(dtype.name, shape, dim))

    @pytest.mark.parametrize("dtype", [tp.float16, tp.int8])
    def test_negative_no_casting(self, dtype):
        from tripy import utils
        from tripy.frontend.trace.ops.iota import Iota

        # TODO: update the 'match' error msg when MLIR-TRT fixes dtype constraint
        out = Iota.build([], dim=0, shape=utils.to_dims((2, 2)), dtype=dtype)
        with helper.raises(
            tp.TripyException,
            match="InternalError: failed to run compilation pipeline",
        ):
            print(out)
