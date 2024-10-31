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
from typing import Sequence

from tripy import constraints, export
from tripy.common.exception import raise_error
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Concatenate(BaseTraceOp):
    dim: int

    def infer_devices(self):
        self.outputs[0].device = self.inputs[0].device

    def infer_len(self):
        # for shapes, only have to sum the input shapes
        from tripy.frontend.trace.ops import utils as op_utils

        out_length = sum(map(lambda inp: op_utils.get_trace_shape(inp)[0], self.inputs))
        return [out_length]

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ConcatenateOp

        if self.dim < 0:
            self.dim += inputs[0].rank
        ConcatenateOp.build(inputs, outputs, dim=self.dim)


@export.public_api(document_under="operations/functions")
@constraints.dtypes(
    constraints={"tensors": "T1", constraints.RETURN_VALUE: "T1"},
    variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
    },
)
def concatenate(tensors: Sequence["tripy.Tensor"], dim: int) -> "tripy.Tensor":
    r"""
    Returns a copy of the input tensor on the target device.

    Args:
        tensors: Sequence of tensors of the same type and having the same shape except in the concatenated dimension.
        dim: the dimension over which the tensors are concatenated.

    Returns:
        Concatenated tensor with shape along `dim` axis equal to sum of dimensions at `dim` axis for all inputs.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.iota((1, 2), dtype=tp.float32)
        b = tp.iota((2, 2), dtype=tp.float32)

        output = tp.concatenate([a, b], dim=0)

        assert np.array_equal(cp.from_dlpack(output).get(), np.concatenate((cp.from_dlpack(a).get(), cp.from_dlpack(b).get()), axis=0))
    """
    if not tensors:
        raise_error(f"Expected a non-empty list of tensors, got {tensors}")

    if len(tensors) == 1:
        return tensors[0]

    return Concatenate.build(list(tensors), dim)
