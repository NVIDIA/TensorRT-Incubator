from dataclasses import dataclass

from tripy import export
from tripy.common import datatype
from tripy.frontend.module.module import Module
from tripy.frontend.module.parameter import Parameter


@export.public_api(document_under="modules")
@dataclass
class Embedding(Module):
    """
    A lookup table for embedding vectors of a fixed size.
    Embedding vectors can be retrieved by their indices.
    """

    dtype: datatype.dtype
    r"""The data type used to perform the operation"""

    weight: Parameter
    r"""The embedding lookup table of shape :math:`[\text{num_embeddings}, \text{embedding_dim}]`."""

    def __init__(self, num_embeddings: int, embedding_dim: int, dtype: datatype.dtype = datatype.float32) -> None:
        r"""
        Args:
            num_embeddings: Number of embedding vectors in the lookup table.
            embedding_dim: Size of each embedding vector in the lookup table.
            dtype: The data type to use for the weight parameter.

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

        self.dtype = dtype

        self.weight = Parameter(iota((num_embeddings, embedding_dim), dtype=dtype))

    def __call__(self, x: "tripy.Tensor") -> "tripy.Tensor":
        r"""
        Args:
            x: A tensor of shape :math:`[N]` containing the indices of the desired embedding vectors.

        Returns:
            A tensor of shape :math:`[N, \text{embedding_dim}]` containing the embedding vectors.
        """
        from tripy.frontend.trace.ops.gather import gather

        return gather(self.weight, 0, x)
