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
from typing import Optional, Sequence

from nvtripy.common.exception import raise_error
from nvtripy.frontend.ops import utils as op_utils


def conv_deconv_helper(
    OpType: type,
    input: "nvtripy.Tensor",
    weight: "nvtripy.Tensor",
    bias: Optional["nvtripy.Tensor"],
    stride: Sequence[int],
    padding: Sequence[Sequence[int]],
    groups: int,
    dilation: Sequence[int],
):
    from nvtripy.frontend.ops.squeeze import squeeze
    from nvtripy.frontend.ops.unsqueeze import unsqueeze

    if input.rank != weight.rank:
        raise_error(
            f"Input and weight should have the same number of spatial dimensions.",
            [f"Input has: {input.rank - 2} spatial dimensions, while weight has: {weight.rank - 2}"],
        )

    stride = list(stride)
    padding = list(padding)
    dilation = list(dilation)

    # Support 1D convolution by unsqueezing
    is_1D = input.rank == 3
    if is_1D:
        input = unsqueeze(input, -1)
        weight = unsqueeze(weight, -1)
        padding.append((0, 0))
        stride.append(1)
        dilation.append(1)

    inputs = [input, weight]
    if bias is not None:
        inputs.append(bias)
    pre_padding, post_padding = op_utils.transform_conv_pooling_padding(padding)
    out = op_utils.create_op(OpType, inputs, stride, pre_padding, post_padding, groups, dilation, stack_depth_offset=1)

    if is_1D:
        out = squeeze(out, -1)
    return out
