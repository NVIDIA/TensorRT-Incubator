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
from dataclasses import dataclass

from mlir_tensorrt.compiler.dialects import tensorrt
from nvtripy.common import datatype
from nvtripy.trace.ops.base import TraceOp


def make_top_k_op(name, attr_name):
    @dataclass(repr=False)
    class TopKOp(TraceOp):
        dim: int
        k: int

        def infer_rank(self):
            rank = self.inputs[0].rank
            self.outputs[0].rank = rank
            self.outputs[1].rank = rank

        def infer_dtypes(self):
            self.outputs[0].dtype = self.inputs[0].dtype
            self.outputs[1].dtype = datatype.int32

        def get_num_outputs(self):
            return 2

        def to_mlir(self, inputs, outputs):
            return tensorrt.top_k(inputs[0], self.k, self.dim, tensorrt.TopKOperationAttr.get(attr_name))

    TopKOp.__name__ = name
    return TopKOp


TopKMax = make_top_k_op("TopKMax", "kMAX")
TopKMin = make_top_k_op("TopKMin", "kMIN")
