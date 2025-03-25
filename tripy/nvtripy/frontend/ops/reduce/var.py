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
import math
from typing import Optional, Sequence, Union

from nvtripy import export
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16"]},
)
def var(
    input: "nvtripy.Tensor", dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False, correction: int = 1
) -> "nvtripy.Tensor":
    r"""
    Returns a new tensor containing the variance of the elements of the input tensor along the specified dimension.

    The variance along a dimension is defined as:

    :math:`\sigma^2 = \Large \frac{1}{max(0, N - \text{correction})} \large \sum_{i=1}^N (x_i - \bar{x})^2`

    where :math:`N` is the length of the dimension, :math:`x_i` is the :math:`i^{th}` element along the dimension,
    and :math:`\bar{x}` is the mean.

    Args:
        input: The input tensor.
        dim: The dimension or dimensions along which to reduce.
            If this is not provided, all dimensions are reduced.
        keepdim: Whether to retain reduced dimensions in the output.
            If this is False, reduced dimensions will be squeezed.
        correction: Defaults to Bessel's correction.

    Returns:
        variance of the input tensor

    .. code-block:: python
        :linenos:

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (2, 3))
        output = tp.var(input, dim=1, keepdim=True)

        torch_input = torch.arange(6, dtype=torch.float32).reshape((2, 3)) # doc: omit
        assert np.array_equal(cp.from_dlpack(output).get(), np.from_dlpack(torch_input.var(dim=1, keepdim=True)))
    """
    from nvtripy.frontend import Tensor
    from nvtripy.frontend.ops.binary.maximum import maximum
    from nvtripy.frontend.ops.cast import cast
    from nvtripy.frontend.ops.reduce.mean import mean
    from nvtripy.frontend.ops.reduce.sum import sum

    dim = op_utils.process_dim_sequence(dim, input.rank)

    mean_val = mean(input, dim=dim, keepdim=True)
    sub = (input - mean_val) ** 2.0

    sum_val = sum(sub, dim=dim, keepdim=keepdim)

    # compute number of elements in the array and divide by number of elements in dims
    shape = sub.shape
    num_elements = math.prod([shape[d] for d in dim])

    num_elements = maximum(num_elements - Tensor(correction), Tensor(0))

    num_elements = cast(num_elements, sum_val.dtype)
    return sum_val / num_elements
