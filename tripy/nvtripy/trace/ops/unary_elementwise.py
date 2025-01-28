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

import enum
from dataclasses import dataclass

import nvtripy.trace.ops.utils as op_utils
from nvtripy.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class UnaryElementwise(BaseTraceOp):
    class Kind(enum.Enum):
        EXP = 0
        TANH = 1
        RSQRT = 2
        LOG = 3
        SINE = 4
        COSINE = 5
        SQRT = 6
        ABS = 7

    kind: Kind

    infer_rank = op_utils.InferRankPolicies.same_as_input()

    # Note: shape inputs will fail because the StableHLO implementations of these ops
    # require float inputs but shapes are always int

    def to_flat_ir(self, inputs, outputs):
        from nvtripy.flat_ir.ops import AbsOp, CosineOp, ExpOp, LogOp, RsqrtOp, SineOp, SqrtOp, TanhOp

        OpType = {
            UnaryElementwise.Kind.EXP: ExpOp,
            UnaryElementwise.Kind.TANH: TanhOp,
            UnaryElementwise.Kind.RSQRT: RsqrtOp,
            UnaryElementwise.Kind.LOG: LogOp,
            UnaryElementwise.Kind.SINE: SineOp,
            UnaryElementwise.Kind.COSINE: CosineOp,
            UnaryElementwise.Kind.SQRT: SqrtOp,
            UnaryElementwise.Kind.ABS: AbsOp,
        }[self.kind]
        OpType.build(inputs, outputs)
