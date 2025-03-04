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
from nvtripy.common import datatype


# Like op_utils.create_op except also performs rank expansion of operands if needed.
# If `cast_bool_to_int` is True, then any boolean inputs are converted to integers before
# performing the operation and then casted back afterwards.
def create_binary_op(OpType, lhs, rhs, cast_bool_to_int: bool = True):
    from nvtripy.frontend.ops.cast import cast

    lhs, rhs = op_utils.match_ranks(lhs, rhs)

    assert lhs.dtype == rhs.dtype, "This function is only implemented for operands with matching data types"
    inp_is_bool = lhs.dtype == datatype.bool
    if cast_bool_to_int and inp_is_bool:
        lhs = cast(lhs, datatype.int8)
        rhs = cast(rhs, datatype.int8)

    out = op_utils.create_op(OpType, [lhs, rhs], stack_depth_offset=1)
    if cast_bool_to_int and inp_is_bool:
        out = cast(out, datatype.bool)
    return out
