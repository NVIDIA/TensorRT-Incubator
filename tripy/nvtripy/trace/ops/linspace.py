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

from dataclasses import dataclass

from mlir_tensorrt.compiler.dialects import tensorrt
from nvtripy.common import datatype
from nvtripy.trace.ops import utils as op_utils
from nvtripy.trace.ops.base import TraceOp


@dataclass(repr=False)
class Linspace(TraceOp):
    dtype: datatype.dtype

    # Returns the best data type to use to perform the linspace operation given the desired output data type.
    # You must cast the output of the linspace back to the desired output data type.
    @staticmethod
    def get_closest_dtype(dt):
        return dt if dt in (datatype.float32, datatype.int32, datatype.int64) else datatype.float32

    infer_rank = op_utils.InferRankPolicies.same_as_shape_of_shape_input()

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def to_mlir(self, inputs, outputs):
        assert len(inputs) == 3, "Linspace operation should have exactly 3 inputs!"
        return [tensorrt.linspace(result=outputs[0], shape=inputs[0], start=inputs[1], step=inputs[2])]
