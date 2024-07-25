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

        # (243): Fix tp.iota() for float16 and int8 type.
        with helper.raises_conditionally(
            dtype in [tp.float16, tp.int8],
            tp.TripyException,
            r"'tensorrt.linspace' op result #0 must be 0D/1D/2D/3D/4D/5D/6D/7D/8D tensor of 32-bit float or 32-bit signless integer values, but got",
        ):
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

        # (243): Fix tp.iota() for float16 and int8 type.
        with helper.raises_conditionally(
            dtype in [tp.float16, tp.int8],
            tp.TripyException,
            r"'tensorrt.linspace' op result #0 must be 0D/1D/2D/3D/4D/5D/6D/7D/8D tensor of 32-bit float or 32-bit signless integer values, but got",
        ):
            assert np.array_equal(cp.from_dlpack(output).get(), self._compute_ref_iota(dtype.name, shape, dim))

    @pytest.mark.parametrize("dtype", [tp.float16, tp.int8])
    def test_negative_no_casting(self, dtype):
        from tripy.frontend.trace.ops.iota import Iota

        # TODO: update the 'match' error msg when MLIR-TRT fixes dtype constraint
        a = tp.ones((2, 2))
        out = Iota.build([a.shape], dim=0, output_rank=2, dtype=dtype)
        with helper.raises(
            tp.TripyException,
            match="error: 'tensorrt.linspace' op result #0 must be 0D/1D/2D/3D/4D/5D/6D/7D/8D tensor of 32-bit float or 32-bit signless integer values",
        ):
            print(out)

    def test_iota_from_shape_tensor(self):
        a = tp.ones((2, 2))
        output = tp.iota(a.shape)
        assert np.array_equal(cp.from_dlpack(output).get(), self._compute_ref_iota("float32", (2, 2), 0))

    def test_iota_from_mixed_seqence(self):
        a = tp.ones((2, 2))
        output = tp.iota((3, a.shape[0]))
        assert np.array_equal(cp.from_dlpack(output).get(), self._compute_ref_iota("float32", (3, 2), 0))
