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


# Like op_utils.create_op except also performs rank expansion of operands if needed.
def create_binary_op(OpType, lhs, rhs):
    from nvtripy.frontend.ops.reshape import reshape

    def expand_rank(tensor, max_rank):
        if tensor.rank == max_rank:
            return tensor

        assert tensor.rank < max_rank, "Tensor rank cannot be larger than max rank of operands"
        new_shape = [1] * (max_rank - tensor.rank) + tensor.shape
        return reshape(tensor, new_shape)

    max_rank = max(lhs.rank, rhs.rank)
    lhs = expand_rank(lhs, max_rank)
    rhs = expand_rank(rhs, max_rank)

    return op_utils.create_op(OpType, [lhs, rhs])
