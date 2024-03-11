from tripy.common import datatype
from tripy.frontend.module.module import Module
from tripy.frontend.module.parameter import Parameter
from tripy.utils import export


@export.public_api(document_under="modules")
class Embedding(Module):
    """
    A lookup table for embedding vectors of a fixed size.
    Embedding vectors can be retrieved by their indices.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, dtype: datatype = datatype.float32):
        r"""
        Args:
            num_embeddings: Number of embedding vectors in the lookup table.
            embedding_dim: Size of each embedding vector in the lookup table.

        .. code-block:: python
            :linenos:
            :caption: Example

            embedding = tp.Embedding(num_embeddings=4, embedding_dim=6)

            input = tp.Tensor([0, 2], dtype=tp.int32)
            output = embedding(input)

            assert np.array_equal(output.numpy(), embedding.weight.numpy()[[0,2], :])
        """
        super().__init__()
        from tripy.frontend.trace.ops.iota import iota

        self.weight: Parameter = Parameter(iota((num_embeddings, embedding_dim), dtype=dtype))
        r"""The embedding lookup table of shape :math:`[\text{num_embeddings}, \text{embedding_dim}]`."""

    def __call__(self, x: "tripy.Tensor") -> "tripy.Tensor":
        r"""
        Args:
            x: A tensor of shape :math:`[N]` containing the indices of the desired embedding vectors.

        Returns:
            A tensor of shape :math:`[N, \text{embedding_dim}]` containing the embedding vectors.
        """
        from tripy.frontend.trace.ops.gather import gather

        return gather(self.weight, 0, x)
