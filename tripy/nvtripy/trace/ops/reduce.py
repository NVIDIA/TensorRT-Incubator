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
from typing import Sequence

from nvtripy.common import datatype
from nvtripy.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Reduce(BaseTraceOp):
    class Kind(enum.Enum):
        def __init__(self, op, init_value):
            self.op = op
            self.init_value = init_value

        SUM = "sum", 0
        MAX = "max", 0
        MUL = "mul", 1
        AND = "and", True
        OR = "or", False

    dim: Sequence[int]
    kind: Kind

    def infer_rank(self):
        self.outputs[0].rank = self.inputs[0].rank - len(self.dim)

    def to_flat_ir(self, inputs, outputs):
        from nvtripy.flat_ir.ops import ConstantOp, ReduceOp
        from nvtripy.flat_ir.tensor import FlatIRTensor

        init_value = self.kind.init_value
        init_const = FlatIRTensor.build(
            shape=(),
            rank=0,
            dtype=outputs[0].dtype,
            device=outputs[0].device,
            reason_details=[
                f"create the constant value tensor (containing {init_value}) for the initial value of a '{self.kind.op}' operation"
            ],
        )
        ConstantOp.build([], [init_const], data=init_value)

        ReduceOp.build([inputs[0], init_const], outputs, reduce_mode=self.kind.op, reduce_dims=self.dim)


@dataclass(repr=False)
class ArgMinMax(Reduce):
    class Kind:
        ARG_MAX = "argmax"
        ARG_MIN = "argmin"

    dim: Sequence[int]
    kind: Kind

    def infer_dtypes(self):
        self.outputs[0].dtype = datatype.int32

    def to_flat_ir(self, inputs, outputs):
        from nvtripy.flat_ir.ops import ArgMinMaxOp, ConstantOp
        from nvtripy.flat_ir.tensor import FlatIRTensor

        init_val_const = FlatIRTensor.build(
            shape=[],
            rank=0,
            dtype=inputs[0].dtype,
            device=outputs[0].device,
            reason_details=[f"create the constant value tensor for the initial value of a '{self.kind}' operation"],
        )
        init_idx_const = FlatIRTensor.build(
            shape=[],
            rank=0,
            dtype=outputs[0].dtype,
            device=outputs[0].device,
            reason_details=[
                f"create the constant value tensor for the initial index value of a '{self.kind}' operation"
            ],
        )

        ConstantOp.build([], [init_val_const], data=0)
        ConstantOp.build([], [init_idx_const], data=0)

        ArgMinMaxOp.build(
            [inputs[0], inputs[1], init_val_const, init_idx_const],
            outputs,
            reduce_mode=self.kind,
            reduce_dims=self.dim,
        )
