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

from typing import List, Union
from dataclasses import dataclass

from tripy import export
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Concatenate(BaseTraceOp):
    dim: int

    def infer_devices(self):
        self.outputs[0].device = self.inputs[0].device

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ConcatenateOp

        if self.dim < 0:
            self.dim += inputs[0].rank
        ConcatenateOp.build(inputs, outputs, dim=self.dim)


@export.public_api(document_under="tensor_operations")
def concatenate(tensors: List[Union["tripy.Tensor"]], dim: int) -> "tripy.Tensor":
    r"""
    Returns a copy of the input tensor on the target device.

    Args:
        tensors: List of tensors of the same type and having the same shape except in the concatenated dimension.
        dim: the dimension over which the tensors are concatenated.

    Returns:
        Concatenated tensor with shape along `dim` axis equal to sum of dimensions at `dim` axis for all inputs.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.iota((2, 3), dtype=tp.float32)
        b = tp.iota((4, 3), dtype=tp.float32)

        output = tp.concatenate([a, b], dim=0)

        assert np.array_equal(cp.from_dlpack(output).get(), np.concatenate((cp.from_dlpack(a).get(), cp.from_dlpack(b).get()), axis=0))
    """
    return Concatenate.build(tensors, dim)
