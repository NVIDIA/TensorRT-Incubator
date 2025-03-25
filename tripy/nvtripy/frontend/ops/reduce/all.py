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
from typing import Optional, Sequence, Union

from nvtripy import export
from nvtripy.utils import wrappers
from nvtripy.common import datatype


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["bool"]},
)
def all(
    input: "nvtripy.Tensor", dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False
) -> "nvtripy.Tensor":
    """
    Returns a new tensor containing the logical AND of the elements
    of the input tensor along the specified dimension.

    Args:
        input: The input tensor.
        dim: The dimension or dimensions along which to reduce.
            If this is not provided, all dimensions are reduced.
        keepdim: Whether to retain reduced dimensions in the output.
            If this is False, reduced dimensions will be squeezed.

    Returns:
        A new bool tensor.

    .. code-block:: python
        :linenos:

        input = tp.Tensor([True, True], dtype=tp.bool)
        out = tp.all(input)
        assert bool(out)
    """
    from nvtripy.frontend.ops.reduce.prod import prod
    from nvtripy.frontend.ops.cast import cast

    return cast(prod(cast(input, dtype=datatype.int32), dim, keepdim), dtype=datatype.bool)
