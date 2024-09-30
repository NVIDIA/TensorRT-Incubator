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
from typing import Optional, Sequence, Tuple

from tripy import constraints, export, utils
from tripy.common.exception import raise_error
from tripy.frontend.trace.ops import utils as op_utils
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Pooling(BaseTraceOp):

    class Kind(enum.Enum):
        def __init__(self, op):
            self.op = op

        MAX = "max"
        AVG = "avg"

    kind: Kind
    kernel_dims: Sequence[int]
    stride: Sequence[int]
    padding: Sequence[Tuple[int]]

    infer_shape_output_idxs = op_utils.ShapeOutputIdxPolicies.never_return_shape

    def infer_rank(self):
        self.outputs[0].rank = self.inputs[0].rank

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ConstantOp, ReduceWindowOp
        from tripy.flat_ir.tensor import FlatIRTensor

        init_value = 0
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

        # extend parameters [spatial_dims,] -> [rank(input),]
        extra_dims = inputs[0].rank - len(self.kernel_dims)
        window_dims = [1] * extra_dims + list(self.kernel_dims)
        window_strides = [1] * extra_dims + list(self.stride)
        padding = [(0, 0)] * extra_dims + list(self.padding)

        ReduceWindowOp.build(
            [inputs[0], init_const],
            outputs,
            reduce_mode=self.kind.op,
            window_dims=window_dims,
            window_strides=window_strides,
            padding=padding,
        )


@export.public_api(document_under="operations/functions")
@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "int8"],
    },
    dtype_constraints={"input": "T1", constraints.RETURN_VALUE: "T1"},
)
def maxpool(
    input: "tripy.Tensor",
    kernel_dims: Sequence[int],
    stride: Optional[Sequence[int]] = None,
    padding: Optional[Sequence[Tuple[int]]] = None,
) -> "tripy.Tensor":
    r"""
    Applies a max pooling over the input tensor.

    The output's non-spatial dimensions are the same as input. For each input spatial dimension
    :math:`D_{i}`, the corresponding output dimension will be:

    .. math::
        D_{out_i} = \left\lfloor\frac{D_{i} + \text{padding_before[i]} + \text{padding_after[i]} -
                \text{kernel_dims[i]}}{\text{stride[i]}} + 1\right\rfloor

    Args:
        input: The input tensor.
        kernel_dims: The spatial shape of the pooling window. Only 2-D or 3-D ``kernel_dims`` are supported.
            If the input has ``int8`` datatype, ``kernel_dims`` can only be 2-D.
        stride: A sequence of length :math:`M` indicating the stride of pooling across each spatial dimension,
            where :math:`M` is the number of spatial dimensions, i.e. :math:`M = \text{rank(input)} - 2`.
            Defaults to all 1.
        padding: A sequence of pairs of integers of length :math:`M` indicating the zero padding
            to apply to the input along each spatial dimension before and after the dimension respectively,
            where :math:`M` is the number of spatial dimensions, i.e. :math:`M = \text{rank(input)} - 2`.
            Defaults to all 0.

    Returns:
        The result tensor after the pooling operation.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.reshape(tp.arange(16, dtype=tp.float32), (1, 1, 4, 4))
        output = tp.maxpool(input, kernel_dims=(2, 2))

        pool_torch = torch.nn.MaxPool2d((2, 2), stride=1) # doc: omit
        expected = pool_torch(torch.from_dlpack(input).to("cpu")) # doc: omit

        assert torch.allclose(torch.from_dlpack(output).to("cpu"), expected)
    """
    spatial_dims = len(kernel_dims)
    if spatial_dims != 2 and spatial_dims != 3:
        raise_error("Unsupported kernel_dims, must be 2D or 3D.", [f"Got kernel_dims={kernel_dims}"])

    op_utils.check_conv_pooling_args(kernel_dims, stride, padding)
    stride = utils.default(stride, [1] * spatial_dims)
    padding = utils.default(padding, [(0, 0)] * spatial_dims)

    return Pooling.build([input], Pooling.Kind.MAX, kernel_dims, stride, padding)
