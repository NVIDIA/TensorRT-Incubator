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

from nvtripy.common import datatype
from nvtripy.frontend.ops import utils as op_utils


def reduce_impl(ReduceOpType, input, dim, keepdim):
    from nvtripy.frontend.ops.squeeze import squeeze
    from nvtripy.frontend.ops.unsqueeze import unsqueeze

    dim = op_utils.process_dim_sequence(dim, input.rank)
    # Reductions require at least 1 dimension, so we can unsqueeze if needed.
    needs_unsqueeze = input.rank == 0
    if needs_unsqueeze:
        input = unsqueeze(input, -1)
        dim = [0]

    output = op_utils.create_op(ReduceOpType, [input], dim, keepdim)
    # We only need to squeeze if the reduction kept the reduction dimension.
    if needs_unsqueeze and keepdim:
        output = squeeze(output, -1)

    return output


def topk_impl(TopKType, input, dim, k):
    from nvtripy.frontend.ops.squeeze import squeeze
    from nvtripy.frontend.ops.unsqueeze import unsqueeze
    from nvtripy.frontend.tensor import Tensor

    if input.rank == 0:
        # TODO (#496): Remove this hack of adding 0 when inputs can be returned directly in compiled functions.
        return input + 0, Tensor(0, dtype=datatype.int32)

    dim = op_utils.process_dim(dim, input.rank)

    # Top-K requires 2D inputs, so we must unsqueeze
    should_unsqueeze_1D = input.rank == 1
    if should_unsqueeze_1D:
        input = unsqueeze(input, -1)

    # Create op returns both values and indices
    values, indices = op_utils.create_op(TopKType, [input], dim=dim, k=k)

    if should_unsqueeze_1D:
        values = squeeze(values, -1)
        indices = squeeze(indices, -1)

    return values, indices


def arg_min_max_impl(TopKType, input, dim, keepdim):
    from nvtripy.frontend.ops.reshape import reshape
    from nvtripy.frontend.ops.squeeze import squeeze
    from nvtripy.frontend.tensor import Tensor

    original_rank = input.rank

    if original_rank == 0:
        return Tensor(0, dtype=datatype.int32)

    # The semantics of argmin/argmax are that the input is treated as a
    # flattened tensor if dim is not set, except the output rank should match
    # the input rank if keepdim=True.
    should_flatten = dim is None
    if should_flatten:
        input = reshape(input, (-1,))
        dim = 0

    _, indices = topk_impl(TopKType, input, dim, k=1)

    # Top-k always keeps dimensions; squeeze if keepdim is False.
    if not keepdim:
        indices = squeeze(indices, dim)

    # Only unflatten at the end if we need to keep the dimensions.
    if should_flatten and keepdim:
        indices = reshape(indices, (1,) * original_rank)

    return indices
