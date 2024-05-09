import tripy as tp
from tests import helper
from tripy.frontend.trace.ops import Slice


class TestSlice:
    def test_op_func_all_partial(self):
        a = tp.Tensor([1, 2, 3, 4])
        a = a[:2]
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Slice)

    def test_incorrect_index_size(self):
        a = tp.Tensor([[1, 2], [3, 4]], shape=(2, 2))
        b = a[:, :, 0:1]

        with helper.raises(
            tp.TripyException,
            match=r"Input tensor has a rank of 2 but was attempted to be sliced with 3 indices.",
            has_stack_info_for=[a, b],
        ) as exc:
            b.eval()

    def test_infer_rank(self):
        a = tp.ones((2, 3))
        a = a[:2, :]
        assert a.trace_tensor.rank == 2

    def test_scalar_index(self):
        a = tp.ones((2, 3, 4))
        assert a[0].shape == [3, 4]
        assert a[0:1].shape == [1, 3, 4]
        assert list(a[0].shape.numpy()) == [3, 4]

    def test_tensor_index(self):
        idx = tp.Tensor(1, dtype=tp.int32)
        a = tp.ones((2, 3))
        b = a[idx]
        assert b.shape == [3]

    def test_invalid_index(self):
        a = tp.ones((2, 3, 4))
        with helper.raises(
            tp.TripyException,
            # note that the stack trace includes an ANSI color code before the caret
            # Looks like:
            # |             a[3].eval()
            # |               ^
            match=r"\| {13}a\[3\]\.eval\(\)\n\s*\| {15}\x1b\[38;5;1m\^\n\n",
            has_stack_info_for=[a],
        ):
            a[3].eval()

    def test_invalid_slice(self):
        a = tp.ones((2, 3, 4))
        with helper.raises(
            tp.TripyException,
            # Looking for a match that looks like this:
            # |             a[3:2:1].eval()
            # |               ^
            # for each dimension
            match=(
                r"\| {13}a\[3:2:1\]\.eval\(\)\n\s*\| {15}\x1b\[38;5;1m\^\n\n"
                r"(.|\n)*\| {13}a\[3:2:1\]\.eval\(\)\n\s*\| {17}\x1b\[38;5;1m\^\n\n"
                r"(.|\n)*\| {13}a\[3:2:1\]\.eval\(\)\n\s*\| {19}\x1b\[38;5;1m\^\n\n"
            ),
            has_stack_info_for=[a],
        ):
            a[3:2:1].eval()

    def test_invalid_multiple_dims(self):
        a = tp.ones((2, 3, 4))
        second_dim_regex = r"(.|\n)*\| {13}a\[3:5, 2\]\.eval\(\)\n\s*\| {20}\x1b\[38;5;1m\^\n\n"
        with helper.raises(
            tp.TripyException,
            # Looking the following:
            # |             a[3:5, 2].eval()
            # |               ^
            #
            # |             a[3:5, 2].eval()
            # |                 ^
            #
            # |             a[3:5, 2].eval()
            # |               ^^^ # cannot narrow down to a single entry
            #
            # and three instances of the following:
            # |             a[3:5, 2].eval()
            # |                    ^
            match=(
                r"\| {13}a\[3:5, 2\]\.eval\(\)\n\s*\| {15}\x1b\[38;5;1m\^\n\n"
                r"(.|\n)*\| {13}a\[3:5, 2\]\.eval\(\)\n\s*\| {17}\x1b\[38;5;1m\^\n\n"
                r"(.|\n)*\| {13}a\[3:5, 2\]\.eval\(\)\n\s*\| {15}\x1b\[38;5;1m\^\^\^\n\n" + 3 * second_dim_regex
            ),
            has_stack_info_for=[a],
        ):
            a[3:5, 2].eval()
