#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from dataclasses import dataclass
from typing import List, Sequence, Set

import tripy.common
from tripy import utils
from tripy.common.array import Array
from tripy.frontend.trace.ops.base import BaseTraceOp
import tripy.frontend.trace.ops.utils as op_utils


@dataclass(repr=False)
class Storage(BaseTraceOp):

    data: Array
    shape: Sequence[int]
    dtype: type
    device: tripy.common.device

    def __init__(self, inputs: List["Tensor"], outputs: List["Tensor"], data: Array) -> None:
        super().__init__(inputs, outputs)

        self.data = data
        self.shape = data.shape
        self.dtype = data.dtype
        self.device = data.device

    # for storage, we will always consider the result to be an ordinary tensor
    infer_shape_output_idxs = op_utils.ShapeOutputIdxPolicies.never_return_shape

    def str_skip_fields(self) -> Set[str]:
        if utils.should_omit_constant_in_str(self.shape):
            return {"data"}
        return set()

    def __eq__(self, other) -> bool:
        return self.data == other.data if isinstance(other, Storage) else False

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def infer_rank(self):
        self.outputs[0].rank = len(self.shape)

    def infer_devices(self):
        # This is different from self.device
        # Constants are always on device when executed by mlir
        self.outputs[0].device = tripy.common.device("gpu")

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ConstantOp

        ConstantOp.build(inputs, outputs, data=self.data)
