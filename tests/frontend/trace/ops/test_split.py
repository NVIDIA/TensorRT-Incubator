import tripy as tp
from tests import helper
from tripy.frontend.trace.ops import Split


class TestSplit:
    def test_basic_instance(self):
        t = tp.ones((4, 5, 6))
        sp = tp.split(t, 2)
        assert len(sp) == 2
        assert isinstance(sp, list)
        for i in range(len(sp)):
            assert isinstance(sp[i], tp.Tensor)
            assert isinstance(sp[i].trace_tensor.producer, Split)
            assert sp[i].trace_tensor.rank == 3
            assert sp[i].shape == [2, 5, 6]

    def test_different_axis(self):
        t = tp.ones((4, 5, 6))
        sp = tp.split(t, 3, dim=2)
        assert len(sp) == 3
        assert isinstance(sp, list)
        for i in range(len(sp)):
            assert isinstance(sp[i], tp.Tensor)
            assert isinstance(sp[i].trace_tensor.producer, Split)
            assert sp[i].shape == [4, 5, 3]

    def test_index_list(self):
        t = tp.ones((4, 5, 6))
        sp = tp.split(t, [2, 3], dim=1)
        # :2, 2:3, 3:
        assert len(sp) == 3
        assert isinstance(sp, list)
        expected_shapes = [[2, 5, 6], [1, 5, 6], [3, 5, 6]]
        for i in range(len(sp)):
            assert isinstance(sp[i], tp.Tensor)
            assert isinstance(sp[i].trace_tensor.producer, Split)
            assert sp[i].shape == expected_shapes[i]

    def test_single_slice(self):
        t = tp.ones((2, 2))
        sp = tp.split(t, 1)
        assert isinstance(sp, tp.Tensor)
        assert isinstance(sp.trace_tensor.producer, Split)
        assert sp.shape == [2, 2]

    def test_indivisible_split(self):
        t = tp.ones((2, 2))
        sp = tp.split(t, 3)
        with helper.raises(
            tp.TripyException, match=r"Split input axis 2 must be divisible by the number of sections 3"
        ):
            sp[0].eval()

    def test_indices_out_of_order(self):
        t = tp.ones((5,))
        with helper.raises(
            tp.TripyException, match=r"Split indices must be given in ascending order\, but given \[4, 2, 1\]"
        ):
            sp = tp.split(t, [4, 2, 1])

    def test_empty_indices(self):
        t = tp.ones((5,))
        with helper.raises(tp.TripyException, match=r"Split indices must not be empty"):
            sp = tp.split(t, [])

    def test_zero_splits(self):
        t = tp.ones((5,))
        with helper.raises(tp.TripyException, match=r"Number of sections argument must be positive, but given 0"):
            sp = tp.split(t, 0)

    def test_invalid_split_dimension(self):
        t = tp.ones(
            (5,),
        )
        with helper.raises(tp.TripyException, match=r"Invalid split dimension 2"):
            sp = tp.split(t, 5, dim=2)
