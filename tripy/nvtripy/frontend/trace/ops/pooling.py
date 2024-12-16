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
from typing import Optional, Sequence

from nvtripy import export, utils, wrappers
from nvtripy.common.exception import raise_error
from nvtripy.frontend.trace.ops import utils as op_utils
from nvtripy.frontend.trace.ops.base import BaseTraceOp
import nvtripy.frontend.trace.ops.utils as op_utils


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
    padding: Sequence[Sequence[int]]

    infer_rank = op_utils.InferRankPolicies.same_as_input()

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def to_flat_ir(self, inputs, outputs):
        from nvtripy.flat_ir.ops import ConstantOp, DivideOp, ReduceWindowOp
        from nvtripy.flat_ir.tensor import FlatIRTensor

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

        if self.kind.op == "max":
            ReduceWindowOp.build(
                [inputs[0], init_const],
                outputs,
                reduce_mode=self.kind.op,
                window_dims=window_dims,
                window_strides=window_strides,
                padding=padding,
            )
        elif self.kind.op == "avg":

            reduce_out = FlatIRTensor.build(
                rank=outputs[0].rank,
                dtype=outputs[0].dtype,
                device=outputs[0].device,
                reason_details=[f"create the output of reduce `{self.kind.op}` operation."],
            )

            ReduceWindowOp.build(
                [inputs[0], init_const],
                [reduce_out],
                reduce_mode=self.kind.op,
                window_dims=window_dims,
                window_strides=window_strides,
                padding=padding,
            )

            window_elements = 1
            for dim in window_dims:
                window_elements *= dim

            # window_elements = compute_window_elements(self.kernel_dims, self.padding)
            init_const = FlatIRTensor.build(
                shape=(),
                rank=0,
                dtype=outputs[0].dtype,
                device=outputs[0].device,
                reason_details=[
                    f"create the constant value tensor (containing {window_elements}) for the divisor of average pool operation."
                ],
            )
            ConstantOp.build([], [init_const], data=window_elements)
            with FlatIRTensor.context(
                [f"expand the rank of constant tensor which is the divisor of average pool operation."]
            ):
                init_const = op_utils.expand_rank_of_tensor(init_const, inputs[0].rank)

            with FlatIRTensor.context([f"broadcast the inputs of division operation."]):
                shape_of_input0 = op_utils.get_shape_of_tensor(reduce_out)
                shape_of_input1 = op_utils.get_shape_of_tensor(init_const)

                # Compute element-wise max of input shapes to get the desired output shape.
                output_shape_tensor = op_utils.compute_shape_of_broadcast(
                    shape_of_input0,
                    shape_of_input1,
                    inputs[0].rank,
                    shape1_name=f"the shape of the first input {shape_of_input0}",
                    shape2_name=f"the shape of the second input {shape_of_input1}",
                )

                init_const = op_utils.insert_broadcast(
                    init_const,
                    out_rank=inputs[0].rank,
                    shape_of_target_tensor=output_shape_tensor,
                    tensor_details=f"left operand",
                )

            DivideOp.build([reduce_out, init_const], outputs)


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "int8"]},
)
def maxpool(
    input: "nvtripy.Tensor",
    kernel_dims: Sequence[int],
    stride: Optional[Sequence[int]] = None,
    padding: Optional[Sequence[Sequence[int]]] = None,
) -> "nvtripy.Tensor":
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
            If the input has :class:`int8` datatype, ``kernel_dims`` can only be 2-D.
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


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "int8"]},
)
def avgpool(
    input: "nvtripy.Tensor",
    kernel_dims: Sequence[int],
    stride: Optional[Sequence[int]] = None,
    padding: Optional[Sequence[Sequence[int]]] = None,
) -> "nvtripy.Tensor":
    r"""
    Applies an average pooling over the input tensor.

    The output's non-spatial dimensions are the same as input. For each input spatial dimension
    :math:`D_{i}`, the corresponding output dimension will be:

    .. math::
        D_{out_i} = \left\lfloor\frac{D_{i} + \text{padding_before[i]} + \text{padding_after[i]} -
                \text{kernel_dims[i]}}{\text{stride[i]}} + 1\right\rfloor

    Args:
        input: The input tensor.
        kernel_dims: The spatial shape of the pooling window. Only 2-D or 3-D ``kernel_dims`` are supported.
            If the input has :class:`int8` datatype, ``kernel_dims`` can only be 2-D.
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
        output = tp.avgpool(input, kernel_dims=(2, 2))

        pool_torch = torch.nn.AvgPool2d((2, 2), stride=1) # doc: omit
        expected = pool_torch(torch.from_dlpack(input).to("cpu")) # doc: omit

        assert torch.allclose(torch.from_dlpack(output).to("cpu"), expected)
    """
    spatial_dims = len(kernel_dims)
    if spatial_dims != 2 and spatial_dims != 3:
        raise_error("Unsupported kernel_dims, must be 2D or 3D.", [f"Got kernel_dims={kernel_dims}"])

    op_utils.check_conv_pooling_args(kernel_dims, stride, padding)
    stride = utils.default(stride, [1] * spatial_dims)
    padding = utils.default(padding, [(0, 0)] * spatial_dims)

    return Pooling.build([input], Pooling.Kind.AVG, kernel_dims, stride, padding)
