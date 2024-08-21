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

from tripy import export
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
def copy(input: "tripy.Tensor", device: "tripy.device") -> "tripy.Tensor":
    r"""
    Returns a copy of the input tensor on the target device.

    Args:
        input:
        device: The target device.

    Returns:
        A copy of this tensor on target device.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.Tensor([1, 2], device=tp.device("gpu"))
        output = tp.copy(input, tp.device("cpu"))

        assert np.array_equal(np.from_dlpack(output), np.array([1, 2], dtype=np.float32))
        assert output.trace_tensor.producer.device.kind == "cpu"
    """
    from tripy.frontend.trace.ops import Storage

    if isinstance(input.trace_tensor.producer, Storage) and input.trace_tensor.producer.device == device:
        return input

    return Copy.build([input], device)
