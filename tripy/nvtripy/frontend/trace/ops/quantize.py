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

from nvtripy import export, wrappers
from nvtripy.common import datatype
from nvtripy.frontend.trace.ops import utils as op_utils
from nvtripy.frontend.trace.ops.base import BaseTraceOp

from nvtripy.frontend.trace.ops import utils as op_utils


@dataclass(repr=False)
class Quantize(BaseTraceOp):

    dtype: datatype.dtype
    dim: int

    infer_rank = op_utils.InferRankPolicies.same_as_input()

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def to_flat_ir(self, inputs, outputs):
        from nvtripy.common.datatype import int32
        from nvtripy.flat_ir.ops import (
            ClampOp,
            ConcatenateOp,
            ConvertOp,
            DivideOp,
            DynamicBroadcastOp,
            DynamicReshapeOp,
            RoundNearestEvenOp,
        )
        from nvtripy.flat_ir.tensor import FlatIRTensor

        # Represent quantize as clamp(round((input / scale))) + convert(dtype)
        scaled_tensor = FlatIRTensor.build(
            shape=inputs[0].shape,
            rank=inputs[0].rank,
            dtype=inputs[0].dtype,
            device=inputs[0].device,
            reason_details=["Compute the scaled tensor by dividing input with scale."],
        )
        broadcast_scale = FlatIRTensor.build(
            shape=inputs[0].shape,  # broadcast to input's shape
            rank=inputs[0].rank,
            dtype=inputs[1].dtype,
            device=inputs[1].device,
            reason_details=["Broadcast the scale to the input's shape in quantize operation."],
        )
        if inputs[1].rank == 0 or inputs[1].rank == 1:
            shape_of_input = op_utils.get_shape_of_tensor(inputs[0])
            broadcast_dim = [self.dim] if self.dim is not None else []
            DynamicBroadcastOp.build([inputs[1], shape_of_input], [broadcast_scale], broadcast_dim=broadcast_dim)
        else:
            # block-wise quant, input: [block_size * A, B], scale: [A, B]
            # Broadcast(scale) -> [block_size, A, B]
            # Reshape(scale) -> [block_size * A, B]
            # Divide(input, scale)
            num_blocks = FlatIRTensor.build(
                shape=(1,),
                rank=1,
                dtype=int32,
                device=inputs[0].device,
                reason_details=["Compute the number of blocks in block-wise quantization"],
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

        DivideOp.build([inputs[0], broadcast_scale], [scaled_tensor])

        rounded_tensor = FlatIRTensor.build(
            shape=inputs[0].shape,
            rank=inputs[0].rank,
            dtype=inputs[0].dtype,
            device=inputs[0].device,
            reason_details=["Perform round-to-nearest-even on the scaled tensor in quantize operation"],
        )
        RoundNearestEvenOp.build([scaled_tensor], [rounded_tensor])

        clamped_tensor = FlatIRTensor.build(
            shape=inputs[0].shape,
            rank=inputs[0].rank,
            dtype=inputs[0].dtype,
            device=inputs[0].device,
            reason_details=["Perform clamp on the rounded tensor in quantize operation"],
        )
        clamp_min, clamp_max = op_utils.get_clamp_min_max(inputs[0].dtype, outputs[0].dtype)
        ClampOp.build([clamp_min, rounded_tensor, clamp_max], [clamped_tensor])

        ConvertOp.build([clamped_tensor], outputs)


@export.public_api(document_under="operations/quantization")
@wrappers.interface(
    dtype_constraints={"input": "T1", "scale": "T1", "dtype": "T2", wrappers.RETURN_VALUE: "T2"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16"], "T2": ["int4", "int8", "float8"]},
    convert_to_tensors={"scale"},
)
def quantize(
    input: "nvtripy.Tensor",
    scale: Union["nvtripy.Tensor", numbers.Number, Sequence[numbers.Number], Sequence[Sequence[numbers.Number]]],
    dtype: datatype.dtype,
    dim: Union[int, Any] = None,
) -> "nvtripy.Tensor":
    """
    Quantizes the input Tensor. The valid quantized data types are
    :class:`nvtripy.int8`, :class:`nvtripy.int4`, :class:`nvtripy.float8`.

    If ``dtype`` is :class:`nvtripy.int4`, the result of this function
    cannot be printed as :class:`nvtripy.int4` is an internal quantized
    data type. It must be dequantized :func:`dequantize` to a higher
    precision first.

    If ``dim`` is not given, this function will perform "per-tensor"
    or "block-wise" quantization.

    * For "per-tensor" quantization, the ``scale`` must be a scalar
      tensor or a single python number.

    * For "block-wise" quantization, the ``dtype`` must only be :class:`nvtripy.int4`.
      The ``input`` tensor must only have 2 dimensions, e.g. ``[D0, D1]``.
      The ``scale`` must also be a 2-D tensor or a 2-D python sequence.
      The first dimension of ``scale`` must be able to divide ``D0``,
      where "blocking" is performed. The second dimension of ``scale``
      must equal to ``D1``.

    If ``dim`` is given, this function will perform "per-channel"
    quantization. The ``scale`` must be a 1-D tensor or a python sequence
    both with size of ``input.shape[dim]``.

    Args:
        input: The input tensor.
        scale: The scale tensor. Must be a constant tensor.
        dtype: The quantization data type. Must be a valid quantized data type (see above).
        dim: The dimension for per-channel quantization

    Returns:
        Quantized Tensor.

    .. code-block:: python
        :linenos:
        :caption: Per-tensor quantization

        input = tp.reshape(tp.arange(6, tp.float32), (2, 3))
        scale = 0.99872
        # output = tp.quantize(input, scale, tp.int8)

        # expected = (np.reshape(np.arange(6, dtype=np.float32), (2, 3)) / scale).astype(np.int8) # doc: omit
        # assert np.array_equal(cp.from_dlpack(output).get(), expected)

    .. code-block:: python
        :linenos:
        :caption: Per-channel quantization

        input = tp.Tensor([[0, 1, 2], [3, 4, 5]], dtype=tp.float32)
        scale = [0.99872, 0.96125]
        output = tp.quantize(input, scale, tp.int8, dim=0)

        expected = (np.reshape(np.arange(6, dtype=np.float32), (2, 3)) / np.array(scale).reshape(2, 1)).astype(np.int8) # doc: omit
        assert np.array_equal(cp.from_dlpack(output).get(), expected)

    .. code-block:: python
        :linenos:
        :caption: Block-wise quantization

        # doc: print-locals input, output

        input = tp.Tensor([[0, 1], [2, 3]], dtype=tp.float32)
        scale = [[1.0, 1.0]]
        quant = tp.quantize(input, scale, tp.int4)
        output = tp.dequantize(quant, scale, tp.float32)

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([[0, 1], [2, 3]], dtype=np.float32))

    .. seealso:: :func:`dequantize`
    """
    op_utils.check_qdq_args(input, scale, dtype, dim, True)

    # This is implemented using a special trace op instead of a combination of frontend ops
    # so that it shows up in the trace and can more easily be pattern matched (by defining our
    # own trace op, we have finer control over the generated MLIR).
    return Quantize.build([input, scale], dtype, dim)
