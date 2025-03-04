# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    from nvtripy.frontend.ops.unsqueeze import unsqueeze
    from nvtripy.frontend.ops.squeeze import squeeze

    pre_padding, post_padding = op_utils.transform_conv_pooling_padding(padding)

    # Support 1D convolution by unsqueezing
    is_1D = input.rank == 3
    if is_1D:
        input = unsqueeze(input, -1)
        weight = unsqueeze(weight, -1)

    inputs = [input, weight]
    if bias is not None:
        inputs.append(bias)
    out = op_utils.create_op(OpType, inputs, stride, pre_padding, post_padding, groups, dilation, stack_depth_offset=1)

    if is_1D:
        out = squeeze(out, -1)
    return out
