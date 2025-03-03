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
    lhs, rhs = op_utils.match_ranks(lhs, rhs)
    # TODO (pranavm): Implicit casting for bool inputs? Need to not do that for comparison ops!
    return op_utils.create_op(OpType, [lhs, rhs], stack_depth_offset=1)
