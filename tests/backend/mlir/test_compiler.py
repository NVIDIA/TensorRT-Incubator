from tests import helper
import tripy as tp


# Tests to ensure that we're able to map errors from MLIR-TRT back to the Python code cleanly.
class TestErrorMapping:
    def test_reshape_invalid_volume(self):
        tensor = tp.ones((2, 2))
        reshaped = tp.reshape(tensor, (3, 3))

        with helper.raises(
            tp.TripyException,
            r"error: number of output elements \(9\) doesn't match expected number of elements \(4\)",
            has_stack_info_for=[tensor, reshaped],
        ):
            reshaped.eval()
