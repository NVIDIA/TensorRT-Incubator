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


import nvtripy.common.datatype as tp_dtype
from nvtripy.common.datatype import int32
from nvtripy.common.exception import raise_error


# Creates a Trace operation from the provided frontend tensors and wraps its outputs in frontend Tensors.
def create_op(OpType, inputs, *args, always_cast_to_dimension_size=False, **kwargs):
    from nvtripy.frontend.dimension_size import DimensionSize
    from nvtripy.frontend.tensor import Tensor

    # Operations that operate on only DimensionSize inputs will always yield a DimensionSize.
    # For any mixed operations, DimensionSize must be casted up to Tensor.
    all_inputs_are_dimension_size = all(isinstance(inp, DimensionSize) for inp in inputs)

    def should_cast_to_dimension_size(out):
        return always_cast_to_dimension_size or (all_inputs_are_dimension_size and out.dtype == int32 and out.rank == 0)

    STACK_DEPTH_OF_FROM_TRACE_TENSOR = 4  # Stack depth from API function calls
    op = OpType([inp.trace_tensor for inp in inputs], *args, **kwargs)
    outputs = [
        (
            DimensionSize.from_trace_tensor(out, include_code_index=STACK_DEPTH_OF_FROM_TRACE_TENSOR)
            if should_cast_to_dimension_size(out)
            else Tensor.from_trace_tensor(out, include_code_index=STACK_DEPTH_OF_FROM_TRACE_TENSOR)
        )
        for out in op.outputs
    ]

    if len(outputs) == 1:
        return outputs[0]
    return outputs


def is_minus_one(arg):
    # Avoid doing an == with a Tensor
    return isinstance(arg, int) and arg == -1


def tensor_from_shape_like(arg: "nvtripy.ShapeLike") -> "nvtripy.Tensor":
    from nvtripy.common.datatype import int32
    from nvtripy.frontend.dimension_size import DimensionSize
    from nvtripy.frontend.ops.concatenate import concatenate
    from nvtripy.frontend.ops.reshape import Reshape
    from nvtripy.frontend.tensor import Tensor

    if not arg:
        return Tensor([], dtype=int32)

    concat_tensors = []

    # We accumulate integers so we can create just a single tensor for each contiguous
    # sequence of integers.
    int_buffer = []

    def empty_buffer():
        if not int_buffer:
            return

        concat_tensors.append(Tensor(int_buffer, dtype=int32))
        int_buffer.clear()

    for elem in arg:
        if isinstance(elem, DimensionSize):
            empty_buffer()
            # NOTE: We cannot use the reshape API here since it would lead to an
            # infinite loop when attempting to convert the shape input to a tensor.
            concat_tensors.append(create_op(Reshape, [elem, Tensor([1])], 1))
        else:
            int_buffer.append(elem)

    empty_buffer()

    out = concatenate(concat_tensors, dim=0)
    # We must set the shape of the shape tensor here since otherwise we will not be able
    # to infer ranks in the frontend. Note that the reshape operations above will not result
    # in a tensor with known shapes even though the new shape is actually known.
    out.trace_tensor.shape = [len(arg)]
    return out


# Processes a `dim` (i.e. axis) argument related to a tensor.
# If the dimension is negative, this will convert it to the corresponding positive index.
def process_dim(dim: int, input_rank: int) -> int:
    new_dim = dim
    if dim < 0:
        new_dim = input_rank + dim

    if new_dim < 0 or new_dim >= input_rank:
        raise_error(
            "Dimension argument is out of bounds.",
            [
                f"Note: provided dimension was: {dim}, while the tensor has a rank of: {input_rank}.\n"
                f"Dimension should be in the half-open interval: [{-input_rank}, {input_rank})."
            ],
        )
    return new_dim


##
## Quantize
##

QUANTIZABLE_DTYPES = (tp_dtype.float32, tp_dtype.float16, tp_dtype.bfloat16)
QUANTIZED_DTYPES = (tp_dtype.int8, tp_dtype.int4, tp_dtype.float8)


def is_quantized_dtype(dtype: "nvtripy.common.datatype.dtype") -> bool:
    return dtype in QUANTIZED_DTYPES


def check_qdq_args(input, scale, dtype, dim, is_quantize):
    valid_input_dtypes = QUANTIZABLE_DTYPES if is_quantize else QUANTIZED_DTYPES
    valid_target_dtypes = QUANTIZED_DTYPES if is_quantize else QUANTIZABLE_DTYPES
    op_str = "quantize op" if is_quantize else "dequantize op"

    if input.dtype not in valid_input_dtypes:
        raise_error(
            f"Input does not have a valid dtype in {op_str}.",
            [
                f"input.dtype must be one of {valid_input_dtypes}, ",
                f"Got dtype={input.dtype}",
            ],
        )

    if dtype not in valid_target_dtypes:
        raise_error(
            f"Unsupported dtype in {op_str}.",
            [
                f"Supported dtypes are: {valid_target_dtypes}. ",
                f"Got dtype={dtype}",
            ],
        )

    quantizable_dtype, quantized_dtype = (input.dtype, dtype) if is_quantize else (dtype, input.dtype)
    if scale.dtype != quantizable_dtype:
        raise_error(
            f"Scale dtype does not match expected dtype in {op_str}.",
            [f"scale should have dtype={quantizable_dtype}, got {scale.dtype}"],
        )

    if dim is not None:
        # per-channel
        if scale.rank != 1:
            raise_error(
                f"If dim is given, scale must be a 1-D tensor in per-channel {op_str}.",
                [f"scale has rank={scale.rank}."],
            )
    elif scale.rank == 2:
        # block-wise:
        if input.rank != 2:
            raise_error(
                f"Input must be a 2-D tensor in block-wise {op_str}.",
                [f"input has rank={input.rank}."],
            )
        if quantized_dtype != tp_dtype.int4:
            raise_error(
                f"Unsupported dtype in block-wise {op_str}.",
                [f"Only `tp.int4` is supported, got {quantized_dtype}"],
            )
    elif scale.rank != 0:
        # per-tensor
        raise_error(
            f"Scale must be a scalar tensor in per-tensor {op_str}.",
            [f"scale has rank={scale.rank}."],
        )


##
## Conv & Pooling
##
def check_conv_pooling_args(kernel_dims, stride, padding, dilation=None):
    spatial_dims = len(kernel_dims)

    if stride is not None:
        if len(stride) != spatial_dims:
            raise_error(
                "Stride must have the same length as kernel_dims.",
                [f"Got stride={stride}, ", f"kernel_dims={kernel_dims}"],
            )

        if not all(s > 0 for s in stride):
            raise_error(
                "Non-positive stride is not supported.",
                details=[f"Got stride: {stride} but all values must be integers greater than 0."],
            )

    if padding is not None:
        if len(padding) != spatial_dims:
            raise_error(
                "Padding must have the same length as kernel_dims.",
                [f"Got padding={padding}, ", f"kernel_dims={kernel_dims}"],
            )

        if not all(len(pad) == 2 for pad in padding):
            raise_error(
                f"Padding must be provided as a sequence of pairs of integers.",
                details=[f"Supplied padding attribute: {padding} contains sequences that are not of length 2."],
            )

        if not all(p1 >= 0 and p2 >= 0 for p1, p2 in padding):
            raise_error(
                "Negative padding is not supported.",
                details=[f"Got padding: {padding} but all values must be non-negative integers."],
            )

    if dilation is not None:
        if len(dilation) != spatial_dims:
            raise_error(
                "Dilation must have the same length as kernel_dims.",
                [f"Got dilation={dilation}, ", f"kernel_dims={kernel_dims}"],
            )

        if not all(isinstance(d, int) and d > 0 for d in dilation):
            raise_error(
                "Non-positive dilation is not supported.",
                details=[f"Got dilation: {dilation} but all values must be integers greater than 0."],
            )
