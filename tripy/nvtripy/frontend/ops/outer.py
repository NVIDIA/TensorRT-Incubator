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
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"vec1": "T1", "vec2": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16"]},
)
def outer(vec1: "nvtripy.Tensor", vec2: "nvtripy.Tensor") -> "nvtripy.Tensor":
    r"""
    Computes the outer product of 1D vectors ``vec1`` and ``vec2``, such that the
    output shape is :math:`(m, n)` if the inputs are of size :math:`(m,)` and :math:`(n,)` respectively.

    Args:
        vec1: The first 1d input vector.
        vec2: The second 1d input vector.

    Returns:
        The outer product of the input vectors.

    .. code-block:: python
        :linenos:

        v1 = tp.arange(5, dtype=tp.float32)
        v2 = tp.arange(4, dtype=tp.float32)
        output = tp.outer(v1, v2)

        t1 = torch.arange(5, dtype=torch.float32) # doc: omit
        t2 = torch.arange(4, dtype=torch.float32) # doc: omit
        torch_out = torch.outer(t1, t2) # doc: omit
        assert tp.allclose(output, tp.Tensor(torch_out))
        assert output.shape == tuple(torch_out.shape)
    """
    from nvtripy.common.exception import raise_error
    from nvtripy.frontend.ops.unsqueeze import unsqueeze

    if vec1.rank != 1 or vec2.rank != 1:
        raise_error(
            "Expected input vectors to be 1D.",
            [f"Got vec1.rank={vec1.rank}, ", f"vec2.rank={vec2.rank}"],
        )

    return unsqueeze(vec1, -1) @ unsqueeze(vec2, 0)
