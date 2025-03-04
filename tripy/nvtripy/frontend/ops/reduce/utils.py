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

from nvtripy.frontend.ops import utils as op_utils


def arg_min_max_impl(TopKType, input, dim, keepdim):
    from nvtripy.frontend.ops.reshape import reshape
    from nvtripy.frontend.ops.squeeze import squeeze

    original_rank = input.rank
    should_flatten = dim is None
    if should_flatten:
        input = reshape(input, (1, -1))
        dim = 1

    dim = op_utils.process_dim(dim, input.rank)

    _, indices = op_utils.create_op(TopKType, [input], dim=dim, k=1)

    # Top-k always keeps dimensions
    if not keepdim:
        indices = squeeze(indices, dim)

    if should_flatten:
        indices = reshape(indices, (1,) * original_rank)

    return indices
