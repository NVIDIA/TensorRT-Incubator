from tripy.frontend.nn.module import Module
from tripy.frontend.nn.parameter import Parameter

from tripy.frontend.ops import gather


class Embedding(Module):
    """
    Embedding is a look up table of a fixed size and is used to retrieve the stored vector at a particular index at runtime.

    Example:
    ::

        embedding = tp.nn.Embedding(4, 6) # 4 elements, each of dimension 6
        input = tp.arange(0, 3, dtype=tp.int32)
        out = embedding(input)
        print(out)

        assert out.numpy().shape == (3, 6)
    """

    def __init__(self, num_embeddings, embedding_dim):
        """
        Args:
            num_embeddings: Number of elements in the look-up table.
            embedding_dim: Size of each element (vector dimensionality) in the look-up table.
        """
        super().__init__()
        from tripy.common.datatype import float32
        from tripy.frontend.tensor_ops import ones

        # Replace with random weights when #74 is completed.
        self.weight = Parameter(ones((num_embeddings, embedding_dim), dtype=float32))

    def __call__(self, x):
        return self.weight.gather(0, x)
