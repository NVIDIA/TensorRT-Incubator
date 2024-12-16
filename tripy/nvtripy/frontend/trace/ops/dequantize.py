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

import numbers
from dataclasses import dataclass
from typing import Any, Sequence, Union

import nvtripy.frontend.trace.ops.utils as op_utils
from nvtripy import export, wrappers
from nvtripy.common import datatype
from nvtripy.frontend.trace.ops import utils as op_utils
from nvtripy.frontend.trace.ops.base import BaseTraceOp
import nvtripy.frontend.trace.ops.utils as op_utils


@dataclass(repr=False)
class Dequantize(BaseTraceOp):

    dtype: datatype.dtype
    dim: int

    infer_rank = op_utils.InferRankPolicies.same_as_input()

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def to_flat_ir(self, inputs, outputs):
        from nvtripy.common.datatype import int32
        from nvtripy.flat_ir.ops import ConcatenateOp, ConvertOp, DivideOp, DynamicBroadcastOp, DynamicReshapeOp, MulOp
        from nvtripy.flat_ir.tensor import FlatIRTensor

        # Represent quantize as convert(input, dtype) * scale
        converted_tensor = FlatIRTensor.build(
            shape=inputs[0].shape,
            rank=inputs[0].rank,
            dtype=self.dtype,
            device=inputs[0].device,
            reason_details=["Convert the input tensor to dequantized dtype."],
        )
        ConvertOp.build([inputs[0]], [converted_tensor])

        broadcast_scale = FlatIRTensor.build(
            shape=inputs[0].shape,  # broadcast to input's shape
            rank=inputs[0].rank,
            dtype=inputs[1].dtype,  # original scale's dtype
            device=inputs[1].device,
            reason_details=["Broadcast the scale to the input's shape in dequant operation."],
        )
        if inputs[1].rank == 0 or inputs[1].rank == 1:
            shape_of_input = op_utils.get_shape_of_tensor(inputs[0])
            broadcast_dim = [self.dim] if self.dim is not None else []
            DynamicBroadcastOp.build([inputs[1], shape_of_input], [broadcast_scale], broadcast_dim=broadcast_dim)
        else:
            # block-wise quant, input: [block_size * A, B], scale: [A, B]
            # Broadcast(scale) -> [block_size, A, B]
            # Reshape(scale) -> [block_size * A, B]
            # Mul(input, scale)
            num_blocks = FlatIRTensor.build(
                shape=(1,),
                rank=1,
                dtype=int32,
                device=inputs[0].device,
                reason_details=["Compute the number of blocks in block-wise dequantization"],
            )
            blocked_shape = FlatIRTensor.build(
                shape=(3,),
                rank=1,
                dtype=int32,
                device=inputs[0].device,
                reason_details=["Compute shape with an extra blocked_size dimension."],
            )
            blocked_scale = FlatIRTensor.build(
                rank=3,
                dtype=inputs[1].dtype,
                device=inputs[1].device,
                reason_details=["Construct the scale to have an extra block_size dimension."],
            )

            input_dim0 = op_utils.get_dim_size_1d_tensor(inputs[0], dim=0)
            scale_dim0 = op_utils.get_dim_size_1d_tensor(inputs[1], dim=0)
            feat_dim = op_utils.get_dim_size_1d_tensor(inputs[1], dim=1)
            DivideOp.build([input_dim0, scale_dim0], [num_blocks])
            ConcatenateOp.build([num_blocks, scale_dim0, feat_dim], [blocked_shape], dim=0)
            DynamicBroadcastOp.build([inputs[1], blocked_shape], [blocked_scale], broadcast_dim=[1, 2])
            origin_input_shape = op_utils.get_shape_of_tensor(inputs[0])
            DynamicReshapeOp.build([blocked_scale, origin_input_shape], [broadcast_scale])

        MulOp.build([converted_tensor, broadcast_scale], outputs)


@export.public_api(document_under="operations/quantization")
@wrappers.interface(
    dtype_constraints={"input": "T1", "scale": "T2", "dtype": "T2", wrappers.RETURN_VALUE: "T2"},
    dtype_variables={"T1": ["int4", "int8", "float8"], "T2": ["float32", "float16", "bfloat16"]},
    convert_to_tensors={"scale"},
)
def dequantize(
    input: "nvtripy.Tensor",
    scale: Union["nvtripy.Tensor", numbers.Number, Sequence[numbers.Number], Sequence[Sequence[numbers.Number]]],
    dtype: datatype.dtype,
    dim: Union[int, Any] = None,
) -> "nvtripy.Tensor":
    """
    Dequantizes the input tensor.

    If ``dim`` is not given, this function will perform "per-tensor"
    or "block-wise" dequantization.

    * For "per-tensor" dequantization, the ``scale`` must be a scalar
      tensor or a single python number.

    * For "block-wise" dequantization, the ``dtype`` must only be :class:`nvtripy.int4`.
      The ``input`` tensor must only have 2 dimensions, e.g. ``[D0, D1]``.
      The ``scale`` must also be a 2-D tensor or a 2-D python sequence.
      The first dimension of ``scale`` must be able to divide ``D0``,
      where "blocking" is performed. The second dimension of ``scale``
      must equal to ``D1``.


    If ``dim`` is given, this function will perform "per-channel"
    dequantization. The ``scale`` must be a 1-D tensor or a python sequence
    both with size of ``input.shape[dim]``.

    Args:
        input: The input tensor with a valid quantized data type.
        scale: The scale tensor. Must be a constant tensor.
        dtype: The data type after dequantization. Must be :class:`nvtripy.float32` or :class:`nvtripy.float16`.
        dim: The dimension for per-channel dequantization

    Returns:
        The dequantized tensor.

    .. code-block:: python
        :linenos:
        :caption: Per-tensor dequantization

        input = tp.Tensor([1, 2, 3], dtype=tp.int8)
        scale = 0.99872
        output = tp.dequantize(input, scale, tp.float32)

        expected = (np.array([1, 2, 3], dtype=np.int8) * scale).astype(np.float32) # doc: omit
        assert np.array_equal(cp.from_dlpack(output).get(), expected)

    .. code-block:: python
        :linenos:
        :caption: Per-channel dequantization

        input = tp.Tensor([[1, 2, 3], [4, 5, 6]], dtype=tp.int8)
        scale = [0.99872, 0.96125]
        output = tp.dequantize(input, scale, tp.float32, dim=0)

        expected = (np.array([[1, 2, 3], [4, 5, 6]]) * np.array(scale).reshape(2, 1)).astype(np.float32) # doc: omit
        assert np.array_equal(cp.from_dlpack(output).get(), expected)

    .. code-block:: python
        :linenos:
        :caption: Block-wise dequantization

        # doc: print-locals input, output

        input = tp.Tensor([[0, 1], [2, 3]], dtype=tp.float32)
        scale = [[1.0, 1.0]]
        quant = tp.quantize(input, scale, tp.int4)
        output = tp.dequantize(quant, scale, tp.float32)

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([[0, 1], [2, 3]], dtype=np.float32))

    .. seealso:: :func:`quantize`
    """
    op_utils.check_qdq_args(input, scale, dtype, dim, False)

    # See the note in quantize.py on why we don't just use frontend ops here.
    return Dequantize.build([input, scale], dtype, dim)
