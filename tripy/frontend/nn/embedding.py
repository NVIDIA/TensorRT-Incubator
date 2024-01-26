from tripy.frontend.nn.module import Module
from tripy.frontend.nn.parameter import Parameter


class Embedding(Module):
    """
    A lookup table for embedding vectors of a fixed size.
    Embedding vectors can be retrieved by their indices.
    """

    def __init__(self, num_embeddings, embedding_dim):
        """
        Args:
            num_embeddings: Number of elements in the lookup table.
            embedding_dim: Size of each embedding vector (i.e. dimensionality) in the lookup table.

        Example:

        .. code:: python
            :number-lines:

            embedding = tp.nn.Embedding(num_embeddings=4, embedding_dim=6)

            input = tp.arange(0, 3, dtype=tp.int32)
            out = embedding(input)
            print(out)

            assert np.array_equal(out.numpy(), np.ones((3, 6), dtype=np.float32))
        """
        super().__init__()
        from tripy.common.datatype import float32
        from tripy.frontend.tensor_initializers import ones

        # Replace with random weights when #74 is completed.
        self.weight: "tp.nn.Parameter" = Parameter(ones((num_embeddings, embedding_dim), dtype=float32))
        """The tensor that stores the embedding lookup table"""

    def __call__(self, x):
        return self.weight.gather(0, x)
