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

from typing import Optional, Sequence, Tuple

from nvtripy import export
from nvtripy.common import datatype
from nvtripy.common.exception import raise_error
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.frontend.ops.pooling import utils as pooling_utils
from nvtripy.trace.ops.pooling import AvgPooling
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "bfloat16", "float16", "int8"]},
)
def avgpool(
    input: "nvtripy.Tensor",
    kernel_dims: Sequence[int],
    stride: Optional[Sequence[int]] = None,
    padding: Optional[Sequence[Tuple[int, int]]] = None,
    count_include_pad: bool = True,
) -> "nvtripy.Tensor":
    r"""
    Applies an average pooling over the input tensor.

    The output's non-spatial dimensions are the same as input. For each input spatial dimension
    :math:`D_{i}`, the corresponding output dimension will be:

    .. math::
        D_{out_i} = \left\lfloor\frac{D_{i} + \text{padding_before[i]} + \text{padding_after[i]} -
                \text{kernel_dims[i]}}{\text{stride[i]}} + 1\right\rfloor

    Args:
        input: The input tensor.
        kernel_dims: The spatial shape of the pooling window. Only 2-D or 3-D ``kernel_dims`` are supported.
            If the input has :class:`int8` datatype, ``kernel_dims`` can only be 2-D.
        stride: A sequence of length :math:`M` indicating the stride of pooling across each spatial dimension,
            where :math:`M` is the number of spatial dimensions, i.e. :math:`M = \text{rank(input)} - 2`.
            Defaults to all 1.
        padding: A sequence of pairs of integers of length :math:`M` indicating the zero padding
            to apply to the input along each spatial dimension before and after the dimension respectively,
            where :math:`M` is the number of spatial dimensions, i.e. :math:`M = \text{rank(input)} - 2`.
            Defaults to all 0.
        count_include_pad: Whether to include padded values in the average.
            Must be ``True`` if the input data type is :class:`int8`.

    Returns:
        The result tensor after the pooling operation.

    .. code-block:: python
        :linenos:

        input = tp.reshape(tp.arange(16, dtype=tp.float32), (1, 1, 4, 4))
        output = tp.avgpool(input, kernel_dims=(2, 2))

        pool_torch = torch.nn.AvgPool2d((2, 2), stride=1) # doc: omit
        expected = pool_torch(torch.from_dlpack(input).to("cpu")) # doc: omit

        assert torch.allclose(torch.from_dlpack(output).to("cpu"), expected)
    """
    op_utils.check_conv_pooling_args(kernel_dims, stride, padding)
    if input.dtype == datatype.int8 and not count_include_pad:
        raise_error("`count_include_pad` must be `True` when input datatype is int8.")

    stride, pre_padding, post_padding = pooling_utils.transform_pooling_params(kernel_dims, stride, padding)

    return op_utils.create_op(
        AvgPooling,
        [input],
        kernel_dims,
        stride,
        pre_padding,
        post_padding,
        avg_excludes_padding=not count_include_pad,
    )
