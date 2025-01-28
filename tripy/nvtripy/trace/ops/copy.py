#
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
#

from dataclasses import dataclass

import nvtripy.trace.ops.utils as op_utils
from nvtripy.common.device import device
from nvtripy.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Copy(BaseTraceOp):
    target: device

    infer_rank = op_utils.InferRankPolicies.same_as_input()

    def infer_devices(self):
        self.outputs[0].device = self.target

    def to_flat_ir(self, inputs, outputs):
        from nvtripy.flat_ir.ops import CopyOp

        CopyOp.build(inputs, outputs, target=self.target)
