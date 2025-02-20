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
from dataclasses import dataclass
from typing import Tuple

from nvtripy.trace.ops import utils as op_utils
from nvtripy.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Squeeze(BaseTraceOp):

    dims: Tuple[int]

    def infer_rank(self):
        self.outputs[0].rank = self.inputs[0].rank - len(self.dims)

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def to_flat_ir(self, inputs, outputs):
        from nvtripy.flat_ir.ops import DynamicReshapeOp

        select_indices = [i for i in range(inputs[0].rank) if i not in self.dims]
        input_shape = op_utils.get_shape_of_tensor(inputs[0])
        shape_slice = []
        for index in select_indices:
            shape_slice.append(op_utils.slice_rank1_tensor(input_shape, index, reason_details=""))

        output_shape = (
            op_utils.concatenate_tensors(shape_slice, dim=0)
            if len(shape_slice) > 0
            else op_utils.add_constant_tensor_from_list([], inputs[0].device)
        )

        DynamicReshapeOp.build([inputs[0], output_shape], outputs)
