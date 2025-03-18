#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from dataclasses import dataclass

from nvtripy import export, utils
from nvtripy.common import datatype
from nvtripy.frontend.module.module import Module
from nvtripy.frontend.module.parameter import DefaultParameter
from nvtripy.frontend.tensor import Tensor


@export.public_api(document_under="operations/modules")
@dataclass
@utils.wrappers.constant_fields(["dtype"])
class Embedding(Module):
    """
    A lookup table for embedding vectors of a fixed size.
    Embedding vectors can be retrieved by their indices.
    """

    dtype: datatype.dtype
    r"""The data type used to perform the operation"""

    weight: Tensor
    r"""The embedding lookup table of shape :math:`[\text{num_embeddings}, \text{embedding_dim}]`."""

    def __init__(self, num_embeddings: int, embedding_dim: int, dtype: datatype.dtype = datatype.float32) -> None:
        r"""
        Args:
            num_embeddings: Number of embedding vectors in the lookup table.
            embedding_dim: Size of each embedding vector in the lookup table.
            dtype: The data type to use for the weight parameter.

        .. code-block:: python
            :linenos:

            embedding = tp.Embedding(num_embeddings=4, embedding_dim=6)

            embedding.weight = tp.iota(embedding.weight.shape)

            input = tp.Tensor([0, 2], dtype=tp.int32)
            output = embedding(input)

            assert np.array_equal(cp.from_dlpack(output).get(), cp.from_dlpack(embedding.weight).get()[[0,2], :])
        """
        super().__init__()

        self.dtype = dtype

        self.weight = DefaultParameter((num_embeddings, embedding_dim), dtype)

    def forward(self, x: "nvtripy.Tensor") -> "nvtripy.Tensor":
        r"""
        Args:
            x: A tensor of shape :math:`[N]` containing the indices of the desired embedding vectors.

        Returns:
            A tensor of shape :math:`[N, \text{embedding_dim}]` containing the embedding vectors.
        """
        from nvtripy.frontend.ops.gather import gather

        return gather(self.weight, 0, x)
