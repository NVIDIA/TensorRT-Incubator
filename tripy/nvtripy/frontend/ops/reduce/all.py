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
from nvtripy.common import datatype as dt
from nvtripy.frontend.constraints import GetInput, GetReturn
from nvtripy.frontend import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    input_requirements=GetInput("input").dtype == dt.bool,
    output_guarantees=GetReturn(0).dtype == dt.bool,
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

        input = tp.Tensor([True, True])
        out = tp.all(input)
        assert bool(out)
    """
    from nvtripy.frontend.ops.reduce.prod import prod
    from nvtripy.frontend.ops.cast import cast
    from nvtripy.common.exception import raise_error

    # Validate that input is bool - constraint system has already checked this
    # but we need to enforce it at runtime when validation is disabled
    if input.dtype != dt.bool:
        raise_error(
            f"Input must have bool dtype for all(), but got {input.dtype}.",
            [
                "This function only accepts bool tensors. ",
                "Note: If you need to check if all elements are non-zero, first compare with zero: ",
                "tp.all(input != 0)",
            ],
        )

    # Cast to int32 since prod doesn't accept bool, then cast back to bool
    return cast(prod(cast(input, dtype=dt.int32), dim, keepdim), dtype=dt.bool)
