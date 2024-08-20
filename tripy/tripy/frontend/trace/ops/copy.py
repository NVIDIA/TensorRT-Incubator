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

from tripy import export, constraints
from tripy.common.device import device
from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy.frontend.trace.ops.utils import InferLenPolicies


@dataclass(repr=False)
class Copy(BaseTraceOp):
    target: device

    def infer_devices(self):
        self.outputs[0].device = self.target

    infer_len = InferLenPolicies.infer_same_as_first_input

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import CopyOp

        CopyOp.build(inputs, outputs, target=self.target)


@export.public_api(document_under="operations/functions")
@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int8", "int32", "int64", "bool"],
    },
    dtype_constraints={"input": "T1", constraints.RETURN_VALUE: "T1"},
)
def copy(input: "tripy.Tensor", device: "tripy.device") -> "tripy.Tensor":
    r"""
    Returns a copy of the input tensor on the target device.

    Args:
        input: Tensor that will be copied
        device: The target device.

    Returns:
        A copy of input tensor on target device.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.Tensor([1, 2], device=tp.device("gpu"))
        output = tp.copy(input, tp.device("cpu"))

        assert tp.array_equal(output, tp.Tensor([1, 2], dtype=tp.float32))
        assert output.trace_tensor.producer.device.kind == "cpu"
    """

    return Copy.build([input], device)
