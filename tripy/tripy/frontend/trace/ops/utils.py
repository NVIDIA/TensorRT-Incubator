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

from typing import Any, List, Optional, Sequence

import tripy.common.datatype as tp_dtype
from tripy import utils
from tripy.backend.mlir.memref import create_memref
from tripy.common.datatype import bool as tp_bool
from tripy.common.datatype import int32
from tripy.common.device import device
from tripy.common.exception import raise_error
from tripy.flat_ir.ops import (
    CompareOp,
    ConcatenateOp,
    ConstantOp,
    DynamicReshapeOp,
    DynamicSliceOp,
    GetDimensionSizeOp,
    SelectOp,
)
from tripy.flat_ir.tensor import FlatIRTensor
from tripy.utils import Result


# Utility for error messages in wrap_shape_inputs
def write_shape_input_indices_message(inputs: List["tripy.Tensor"]) -> str:
    from tripy.frontend.shape import Shape

    shape_indices = list(map(str, filter(lambda i: isinstance(inputs[i], Shape), range(len(inputs)))))
    if not shape_indices:
        return ""
    if len(shape_indices) == 1:
        return f"input with index {shape_indices[0]} is tp.Shape"
    return f"inputs with indices {', '.join(shape_indices)} are tp.Shape"


def get_broadcast_dim(dim1, dim2):
    if dim1.is_dynamic_dim():
        return dim1
    elif dim2.is_dynamic_dim():
        return dim2
    else:
        assert dim1 == 1 or dim2 == 1 or dim1 == dim2
        # can't just return max(dim1, dim2) because one may be 0
        if dim1 == 1:
            return dim2
        # dim1 == dim2 or dim2 == 1
        return dim1


##
## Handling shape outputs: These are common policies to use for overring infer_shape_output_idxs
##


class ShapeOutputIdxPolicies:
    def infer_from_first_input_only(self, inputs):
        """
        Common override for `infer_shape_output_idxs`: Treat the outputs as shapes if the *first* input is a shape.
        """
        from tripy.frontend.shape import Shape

        if isinstance(inputs[0], Shape):
            return Result.ok({"shape": list(range(len(self.outputs)))})
        return Result.ok({})

    def never_return_shape(self, inputs):
        """
        Accepts shapes but the result is always no shape indices
        """
        return Result.ok({})


##
## Inferring shape lengths (helpers)
##


def get_trace_shape(input: "TraceTensor") -> Sequence[int]:
    """
    Given an operator input tensor, return its shape if it has already been given
    or get its shape from the shape context if it's needed.
    """
    if input.shape is None:
        from tripy.backend.mlir.utils import ShapeContext

        # memoize while we're at it
        input.shape = ShapeContext().get_shape_of_dynamic_trace_tensor(input)
    return input.shape


class InferLenPolicies:
    def infer_same_as_first_input(self):
        return [get_trace_shape(self.inputs[0])[0]]


##
## Helpers
##


def get_dim_size_1d_tensor(tensor: "FlatIRTensor", dim: int):
    # GetDimensionSizeOp returns a scalar
    dim_scalar = FlatIRTensor.build(
        shape=(),
        rank=0,
        dtype=int32,
        device=tensor.device,
        reason_details=[f"Get size of dim {dim}."],
    )
    GetDimensionSizeOp.build([tensor], [dim_scalar], dim=dim)
    # reshape scalar to rank 1
    dim_tensor = reshape_scalar_to_1d(dim_scalar)
    return dim_tensor


def get_shape_of_tensor(tensor: "FlatIRTensor", out: "FlatIRTensor" = None):
    if tensor.rank > 0:
        inp_rank = tensor.rank
        dim_sizes = [None] * inp_rank
        for i in range(inp_rank):
            dim_sizes[i] = get_dim_size_1d_tensor(tensor, i)
        shape_output_tensor = concatenate_tensors(dim_sizes, 0, out)
    else:
        # TODO #111: Remove this codepath when shape dialect is used (shape.shape_of).
        shape_output_tensor = (
            FlatIRTensor.build(
                shape=(),
                rank=1,
                dtype=int32,
                device=tensor.device,
                reason_details=["retrieve the shape of: ", tensor],
            )
            if out is None
            else out
        )
        ConstantOp.build(
            [],
            [shape_output_tensor],
            data=create_memref(shape=(0,), dtype=int32, device=tensor.device),
        )
    return shape_output_tensor


def add_constant_tensor_from_list(data: list, device: "tripy.device"):
    const_output_tensor = FlatIRTensor.build(
        shape=[len(data)],
        rank=1,
        dtype=int32,
        device=device,
        reason_details=[f"create constant rank 1 int32 tensor filled with {data}."],
    )
    if not data:
        data = create_memref(shape=(0,), dtype=int32)
    ConstantOp.build([], [const_output_tensor], data=data)
    return const_output_tensor


def concatenate_tensors(inputs: List["FlatIRTensor"], dim: int, out: Optional["FlatIRTensor"] = None):
    if out is None:
        out = FlatIRTensor.build(
            rank=1,
            dtype=int32,
            device=inputs[0].device,
            reason_details=[
                "output of concatenation of the following tensors: ",
                *[inp for inp in inputs],
                f" along dim {dim}.",
            ],
        )
    ConcatenateOp.build(inputs, [out], dim=dim)
    return out


def reshape_scalar_to_1d(input: "FlatIRTensor"):
    shape_1d = add_constant_tensor_from_list([1], input.device)
    out = FlatIRTensor.build(
        shape=(1,),
        rank=1,
        dtype=int32,
        device=input.device,
        reason_details="Reshape input scalar to 1D.",
    )

    DynamicReshapeOp.build([input, shape_1d], [out])
    return out


##
## Broadcasting
##


def get_broadcast_compatible_shapes(shape1, shape2):
    # Make the shorter shape the same length as the longer shape by padding with ones
    if len(shape1) > len(shape2):
        shape2 = (1,) * (len(shape1) - len(shape2)) + shape2
    elif len(shape2) > len(shape1):
        shape1 = (1,) * (len(shape2) - len(shape1)) + shape1

    return shape1, shape2


def is_broadcast_compatible(shape1, shape2) -> Result:
    # Now check each dimension pair
    for index, (dim1, dim2) in enumerate(zip(shape1, shape2)):
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            return Result.err(
                [
                    f"for tensor shapes: {shape1} and {shape2}, dimensions on axis {index}: '{dim1}' and '{dim2}' are not broadcast compatible"
                ],
            )

    return Result.ok()


# Given two shapes, compute the shape of the resulting broadcast. Assumes that the shapes are of equal rank
def compute_shape_of_broadcast(
    shape1, shape2, output_rank: int, shape1_name: Optional[str] = None, shape2_name: Optional[str] = None
):
    from tripy.frontend.trace.ops.binary_elementwise import Comparison

    shape1_name = utils.default(shape1_name, "a tensor")
    shape2_name = utils.default(shape2_name, "another tensor")

    # can't just use the max of shape1 and shape2 because it will be incorrect if a dim is 0
    # (the broadcast of 0 and 1 is 0)
    resulting_shape = FlatIRTensor.build(
        shape=[output_rank],
        rank=1,
        dtype=int32,
        device=shape1.device,
        reason_details=[
            f"compute the broadcasted shape of {shape1_name} ",
            shape1,
            f" and {shape2_name} ",
            shape2,
        ],
    )
    shape_dim_comparison = FlatIRTensor.build(
        shape=[output_rank],
        rank=1,
        dtype=tp_bool,
        device=shape1.device,
        reason_details=[
            f"Compare the dims of {shape1_name} with 1",
        ],
    )
    ones = add_constant_tensor_from_list([1] * output_rank, shape1.device)
    # if shape1[i] == 1, use shape2[i]. Otherwise use shape1[i]
    CompareOp.build([shape1, ones], [shape_dim_comparison], compare_direction=Comparison.Kind.EQUAL.compare_direction)
    SelectOp.build([shape_dim_comparison, shape2, shape1], [resulting_shape])
    return resulting_shape


# To which dimension in the target shape each dimension of the operand shape corresponds to.
def get_broadcast_in_dim(input_rank: int, output_rank: int) -> List[int]:
    assert output_rank >= input_rank
    broadcast_dimensions = []
    rank_diff = output_rank - input_rank

    for idx in range(input_rank):
        corresponding_output_dim = idx + rank_diff

        # We might need careful check in case of dynamic dims
        broadcast_dimensions.append(corresponding_output_dim)

    assert len(broadcast_dimensions) == input_rank
    return broadcast_dimensions


# Insert a broadcast op into the flat_ir which broadcasts input tensor to output shape.
# If the output shape is dynamic, shape_of_target_tensr is used to describe the output shape.
# tensor_details should describe what this tensor is (e.g. left operand of '+')
def insert_broadcast(
    input_tensor: "FlatIRTensor",
    out_rank: int,
    shape_of_target_tensor: "FlatIRTensor",
    tensor_details: str,
):
    from tripy.flat_ir.ops import DynamicBroadcastOp
    from tripy.flat_ir.tensor import FlatIRTensor
    from tripy.frontend.trace.ops.utils import get_broadcast_in_dim

    output_tensor = FlatIRTensor.build(
        rank=out_rank,
        dtype=input_tensor.dtype,
        device=input_tensor.device,
        reason_details=[
            f"broadcast the {tensor_details}, which was: ",
            input_tensor,
            f" to a rank of: {out_rank} in order to be compatible with the other input(s)",
        ],
    )

    DynamicBroadcastOp.build(
        [input_tensor, shape_of_target_tensor],
        [output_tensor],
        broadcast_dim=get_broadcast_in_dim(input_tensor.rank, out_rank),
    )

    return output_tensor


# Expands rank of a tensor via prepending extra dims provided by nb_extra_dims.
def expand_rank_of_tensor(input: "FlatIRTensor", nb_extra_dims: int):
    if nb_extra_dims == 0:
        return input

    # Create array filled with 1s and concat with shape array
    assert nb_extra_dims > 0
    # rank 1 tensor
    shape_of_input = get_shape_of_tensor(input)

    # create rank 1 tensor filled with nb_extra_dims
    extra_ones = add_constant_tensor_from_list([1] * nb_extra_dims, input.device)

    concat_output_tensor = FlatIRTensor.build(
        rank=1,
        dtype=int32,
        device=input.device,
        reason_details=[
            f"append {nb_extra_dims} ones to the input shape {shape_of_input} to expand the rank of tensor."
        ],
    )
    ConcatenateOp.build([extra_ones, shape_of_input], [concat_output_tensor], dim=0)

    # output shape usage just relies on rank.
    output_rank = input.rank + nb_extra_dims
    return insert_broadcast(
        input,
        out_rank=output_rank,
        shape_of_target_tensor=concat_output_tensor,
        tensor_details="",
    )


##
## Slice
##


def slice_rank1_tensor(rank1_tensor: "FlatIRTensor", slice_index: int, reason_details: Optional[List[Any]] = None):
    """
    Slice a rank 1 tensor tensor along a certain index.
    Ex: tensor [1,2,3,4,5,6] sliced at slice_index 2 will return 3.
    """

    device = rank1_tensor.device
    start_idx = add_constant_tensor_from_list([slice_index], device)
    stride_index = add_constant_tensor_from_list([1], device)
    slice_len = add_constant_tensor_from_list([slice_index + 1], device)
    result_slice = FlatIRTensor.build(
        rank=1,
        dtype=int32,
        device=device,
        reason_details=reason_details if reason_details is not None else [],
    )
    DynamicSliceOp.build([rank1_tensor, start_idx, slice_len, stride_index], [result_slice])
    return result_slice


##
## Quantize
##

QUANTIZABLE_DTYPES = (tp_dtype.float32, tp_dtype.float16, tp_dtype.bfloat16)
QUANTIZED_DTYPES = (tp_dtype.int8, tp_dtype.int4, tp_dtype.float8)


def is_quantized_dtype(dtype: "tripy.common.datatype.dtype") -> bool:
    return dtype in QUANTIZED_DTYPES


def is_quantizable_dtype(dtype: "tripy.common.datatype.dtype") -> bool:
    return dtype in QUANTIZABLE_DTYPES


def get_clamp_min_max(element_dtype, quant_dtype):
    QUANT_CLAMP_MIN_MAX = {
        tp_dtype.int8: (-128.0, 127.0),
        tp_dtype.int4: (-8.0, 7.0),
        tp_dtype.float8: (-448.0, 448.0),
    }
    min_val, max_val = QUANT_CLAMP_MIN_MAX[quant_dtype]
    clamp_min = FlatIRTensor.build(
        shape=(),
        rank=0,
        dtype=element_dtype,
        device=device("gpu"),
        reason_details=["Construct min value for clamp."],
    )
    clamp_max = FlatIRTensor.build(
        shape=(),
        rank=0,
        dtype=element_dtype,
        device=device("gpu"),
        reason_details=["Construct max value for clamp."],
    )
    ConstantOp.build([], [clamp_min], data=min_val)
    ConstantOp.build([], [clamp_max], data=max_val)
    return clamp_min, clamp_max


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
