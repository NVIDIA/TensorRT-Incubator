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


def reduce_impl(ReduceOpType, input, dim, keepdim):
    if input.rank == 0:
        return input

    # TODO (pranavm): Test all reduction functions with scalars.
    dim = op_utils.process_dim_sequence(dim, input.rank)
    return op_utils.create_op(ReduceOpType, [input], dim, keepdim)


def arg_min_max_impl(TopKType, input, dim, keepdim):
    from nvtripy.frontend.ops.reshape import reshape
    from nvtripy.frontend.ops.squeeze import squeeze
    from nvtripy.frontend.ops.unsqueeze import unsqueeze
    from nvtripy.frontend.tensor import Tensor

    original_rank = input.rank

    if original_rank == 0:
        return Tensor(0, dtype=input.dtype)

    # The semantics of argmin/argmax are that the input is treated as a
    # flattened tensor if dim is not set, except the output rank should match
    # the input rank if keepdim=True.
    should_flatten = dim is None
    if should_flatten:
        input = reshape(input, (-1,))
        dim = 0

    dim = op_utils.process_dim(dim, input.rank)

    # Top-K requires 2D inputs, so we must unsqueeze
    should_unsqueeze_1D = input.rank == 1
    if should_unsqueeze_1D:
        input = unsqueeze(input, -1)

    _, indices = op_utils.create_op(TopKType, [input], dim=dim, k=1)

    if should_unsqueeze_1D:
        indices = squeeze(indices, -1)

    # Top-k always keeps dimensions
    if not keepdim:
        indices = squeeze(indices, dim)

    # Only unflatten at the end if we need to keep the dimensions.
    if should_flatten and keepdim:
        indices = reshape(indices, (1,) * original_rank)

    return indices
