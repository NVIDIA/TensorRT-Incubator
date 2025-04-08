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


from nvtripy import export
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.trace.ops.gather import Gather
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", "index": "T2", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "int4", "int8", "int32", "int64", "bool"],
        "T2": ["int32", "int64"],
    },
)
def gather(input: "nvtripy.Tensor", dim: int, index: "nvtripy.Tensor") -> "nvtripy.Tensor":
    """
    Gather values from the input tensor along the specified axis based on the specified indices.
    This behaves similarly to `numpy.take() <https://numpy.org/doc/2.2/reference/generated/numpy.take.html>`_.

    Args:
        input: The input tensor.
        dim: The dimension along which to gather.
        index: A tensor of indices to gather.
            Values along the provided dimension of the input are effectively replaced by the values
            at the specified indices. If the index tensor is multi-dimensional, the values along
            the dimension will be replaced by tensors instead of scalars.

    Returns:
        A new tensor of the same shape along every
        dimension except ``dim``, which will have a size equal to ``index.shape[0]``.

    .. code-block:: python
        :linenos:

        input = tp.iota((3, )) + 4
        index = tp.Tensor([0, 2])
        output = tp.gather(input, dim=0, index=index)

        assert np.array_equal(cp.from_dlpack(output).get(), np.take(cp.from_dlpack(input).get(), np.from_dlpack(index), axis=0))

    .. code-block:: python
        :linenos:
        :caption: Multi-dimensional Input

        input = tp.iota((3, 3)) + 4
        index = tp.Tensor([0, 2])
        output = tp.gather(input, dim=0, index=index)

        assert np.array_equal(cp.from_dlpack(output).get(), np.take(cp.from_dlpack(input).get(), np.from_dlpack(index), axis=0))

    .. code-block:: python
        :linenos:
        :caption: Multi-dimensional Indices

        input = tp.iota((3, )) + 4
        index = tp.Tensor([[0], [2], [1], [2]])
        output = tp.gather(input, dim=0, index=index)

        assert np.array_equal(cp.from_dlpack(output).get(), np.take(cp.from_dlpack(input).get(), np.from_dlpack(index), axis=0))

    .. code-block:: python
        :linenos:
        :caption: Multi-dimensional Input And Indices

        input = tp.iota((3, 3)) + 4
        index = tp.Tensor([[0, 2], [2, 1]])
        output = tp.gather(input, dim=0, index=index)

        assert np.array_equal(cp.from_dlpack(output).get(), np.take(cp.from_dlpack(input).get(), np.from_dlpack(index), axis=0))
    """
    dim = op_utils.process_dim(dim, input.rank)

    return op_utils.create_op(Gather, [input, index], dim)
