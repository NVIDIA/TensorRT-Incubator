#
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
#

from collections.abc import Sequence

from nvtripy.utils import wrappers
from nvtripy.trace.ops.convolution import Convolution


@wrappers.interface(
    dtype_constraints={"input": "T1", "weight": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16"],
    },
)
def convolution(
    input: "nvtripy.Tensor",
    weight: "nvtripy.Tensor",
    padding: Sequence[Sequence[int]],
    stride: Sequence[int],
    groups: int,
    lhs_dilation: Sequence[int],
    rhs_dilation: Sequence[int],
):
    return Convolution.build([input, weight], padding, stride, groups, lhs_dilation, rhs_dilation)
