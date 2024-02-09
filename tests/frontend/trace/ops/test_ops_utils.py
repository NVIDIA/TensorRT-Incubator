import pytest

from tripy.frontend.trace.ops.utils import get_broadcast_in_dim


class TestGetBroadcastInDim:
    @pytest.mark.parametrize(
        "input_shape, output_shape, expected_dim",
        [
            ([2, 2, 3], [2, 2, 3], [0, 1, 2]),  # no broadcast
            ([2, 3], [2, 2, 3], [1, 2]),  # simple broadcast
            ([], [2, 2, 3], []),  # output should be of same rank as input
            ([5], [2, 4, 5], [2]),
        ],
    )
    def test_static_broadcast_in_dim(self, input_shape, output_shape, expected_dim):
        assert get_broadcast_in_dim(input_shape, output_shape) == expected_dim
