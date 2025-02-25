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

from nvtripy.frontend.ops import utils as op_utils
from nvtripy.utils.utils import make_list


def adjust_dim(dim: Optional[Union[int, Sequence[int]]], rank: int):
    if dim is None:
        dim = list(range(rank))
    dim = [op_utils.process_dim(d, rank) for d in make_list(dim)]
    return dim


def arg_min_max_impl(TopKType, input, dim, keepdim):
    from nvtripy.frontend.ops.reshape import reshape
    from nvtripy.frontend.ops.squeeze import squeeze

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
        indices = reshape(indices, (-1,))

    return indices
