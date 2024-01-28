from tripy.frontend.nn.module import Module
from tripy.frontend.nn.parameter import Parameter


class Embedding(Module):
    """
    A lookup table for embedding vectors of a fixed size.
    Embedding vectors can be retrieved by their indices.
    """

    def __init__(self, num_embeddings, embedding_dim):
        r"""
        Args:
            num_embeddings: Number of embedding vectors in the lookup table.
            embedding_dim: Size of each embedding vector in the lookup table.

        Example:

        .. code:: python
            :number-lines:

            embedding = tp.nn.Embedding(num_embeddings=4, embedding_dim=6)

            input = tp.Tensor([0, 1, 2], dtype=tp.int32)
            output = embedding(input)

            assert np.array_equal(output.numpy(), np.ones((3, 6), dtype=np.float32))
        """
        super().__init__()
        from tripy.common.datatype import float32
        from tripy.frontend.ops import ones

        # Replace with random weights when #74 is completed.
        self.weight: Parameter = Parameter(ones((num_embeddings, embedding_dim), dtype=float32))
        r"""The embedding lookup table of shape :math:`[\text{num_embeddings}, \text{embedding_dim}]`."""

    def __call__(self, x):
        r"""
        Args:
            x: A tensor of shape :math:`[N]` containing the indices of the desired embedding vectors.

        Returns:
            A tensor of shape :math:`[N, \text{embedding_dim}]` containing the embedding vectors.
        """
        return self.weight.gather(0, x)
