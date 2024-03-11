import pytest
import tripy as tp
from tests import helper


class TestEmbedding:
    def test_embedding(self):
        embedding = tp.Embedding(20, 30)
        assert isinstance(embedding, tp.Embedding)
        assert embedding.weight.numpy().shape == (20, 30)

    def test_incorrect_input_dtype(self):
        a = tp.ones((2, 3))
        linear = tp.Embedding(4, 16)

        with helper.raises(
            tp.TripyException,
            match="Index tensor for gather operation should be of int32 type.",
            has_stack_info_for=[a],
        ):
            out = linear(a)
