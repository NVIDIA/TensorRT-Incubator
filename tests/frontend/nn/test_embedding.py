import pytest
import tripy as tp


class TestEmbedding:
    def test_embedding(self):
        embedding = tp.nn.Embedding(20, 30)
        assert isinstance(embedding, tp.nn.Embedding)
        assert embedding.weight.numpy().shape == (20, 30)

    def test_incorrect_input_dtype(self):
        a = tp.ones((2, 3))
        linear = tp.nn.Embedding(4, 16)
        out = linear(a)

        with pytest.raises(
            tp.TripyException, match="Index tensor for gather operation should be of int32 type."
        ) as exc:
            out.eval()
        print(str(exc.value))
