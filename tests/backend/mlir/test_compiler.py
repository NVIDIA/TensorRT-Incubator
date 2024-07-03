import pytest
from tests import helper
import tripy as tp


# Tests to ensure that we're able to map errors from MLIR-TRT back to the Python code cleanly.
class TestErrorMapping:
    def test_invalid_slice(self):
        values = tp.Tensor([1, 2, 3])
        sliced = values[4]

        with helper.raises(
            tp.TripyException,
            r"limit index 5 is larger than dimension size 3 in dimension 0",
            has_stack_info_for=[values],
        ):
            sliced.eval()

    @pytest.mark.skip(
        "TODO (#207): MLIR-TRT currently triggers a C-style abort, which we cannot handle. Needs to be fixed in MLIR-TRT."
    )
    def test_reshape_invalid_volume(self):
        tensor = tp.ones((2, 2))
        reshaped = tp.reshape(tensor, (3, 3))

        with helper.raises(
            tp.TripyException,
            r"output_shape is incompatible with input type of operation: input has 4 elements, but output_shape has 9",
            has_stack_info_for=[tensor, reshaped],
        ):
            reshaped.eval()
