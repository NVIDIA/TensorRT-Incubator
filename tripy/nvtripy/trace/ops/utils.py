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

from typing import List

##
## infer_rank helpers
##


class InferRankPolicies:
    # Indicates that the output has not only the same rank but also the same shape as the input.
    def same_shape_as_input(idx=0):
        def impl(self):
            self.outputs[0].shape = self.inputs[idx].shape

        return impl

    def same_as_input(idx=0):
        def impl(self):
            self.outputs[0].rank = self.inputs[idx].rank

        return impl

    def same_as_shape_of_shape_input(idx=0):
        def impl(self):
            assert len(self.inputs[idx].shape) == 1, "Expected this input to be a shape tensor"
            assert isinstance(self.inputs[idx].shape[0], int), "Expected shape tensor length to be known"
            self.outputs[0].rank = self.inputs[idx].shape[0]

        return impl

    def max_of_inputs():
        def impl(self):
            self.outputs[0].rank = max(inp.rank for inp in self.inputs)

        return impl


##
## Broadcasting
##


# To which dimension in the target shape each dimension of the operand shape corresponds to.
def get_broadcast_in_dim(input_rank: int, output_rank: int) -> List[int]:
    assert output_rank >= input_rank
    broadcast_dimensions = []
    rank_diff = output_rank - input_rank

    for idx in range(input_rank):
        corresponding_output_dim = idx + rank_diff

        # We might need careful check in case of dynamic dims
        broadcast_dimensions.append(corresponding_output_dim)

    assert len(broadcast_dimensions) == input_rank
    return broadcast_dimensions
