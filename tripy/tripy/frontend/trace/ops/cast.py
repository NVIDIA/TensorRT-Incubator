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
from tripy import export, constraints
from tripy.frontend import utils as frontend_utils
from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy.frontend.trace.ops.utils import InferLenPolicies


@dataclass(repr=False)
class Cast(BaseTraceOp):
    dtype: "tripy.common.dtype"

    def infer_tensor_variants(self, inputs):
        from tripy.common.datatype import int32
        from tripy.frontend.shape import Shape, ShapeScalar
        from tripy.utils import Result

        # Only still a valid shape if it remains int32
        if self.dtype == int32:
            if isinstance(inputs[0], (Shape, ShapeScalar)):
                return Result.ok([type(inputs[0])])
        return Result.ok([None])

    infer_len = InferLenPolicies.infer_same_as_first_input

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    @frontend_utils.make_function
    def to_flat_ir(self, inputs, outputs):
        from tripy.common.datatype import int32, int64, float32, bool as tp_bool
        from tripy.flat_ir.ops import CompareOp, ConvertOp, ConstantOp, DynamicBroadcastOp
        from tripy.flat_ir.tensor import FlatIRTensor
        import tripy.frontend.trace.ops.utils as op_utils

        # If we need to create a constant (namely for comparing with zero), it has to use one of these dtypes.
        # If the input is not one of these dtypes, the constant needs to be created in one of these and converted.
        DTYPES_FOR_CONSTANTS = {float32, int32, int64}

        convert_input = inputs[0]

        # For conversion to bool, we must compare with 0 since the underlying semantics for StableHLO
        # are to do truncation for conversion to integer types (and bools are i1). This would get
        # unintended results for even numbers, which truncate to 0 in i1.
        if self.dtype == tp_bool:
            # Creating a zero tensor uses the same logic as the zeros_like initializer

            # If the input dtype does not allow directly creating a Tripy array, we have to use another like f32
            # and then cast the zeros tensor.
            zero_dtype = convert_input.dtype if convert_input.dtype in DTYPES_FOR_CONSTANTS else float32
            single_zero = FlatIRTensor.build(
                shape=[],
                rank=0,
                dtype=zero_dtype,
                device=convert_input.device,
                reason_details=["Zero scalar for casting to bool"],
            )
            ConstantOp.build([], [single_zero], data=0)
            zeros_shape = op_utils.get_shape_of_tensor(convert_input)
            zeros = FlatIRTensor.build(
                shape=convert_input.shape,
                rank=convert_input.rank,
                dtype=zero_dtype,
                device=convert_input.device,
                reason_details=["Tensor of zeroes for comparing to cast to bool"],
            )
            DynamicBroadcastOp.build([single_zero, zeros_shape], [zeros], broadcast_dim=[])

            if zero_dtype != convert_input.dtype:
                zero_output = FlatIRTensor.build(
                    shape=zeros.shape,
                    rank=zeros.rank,
                    dtype=convert_input.dtype,
                    device=zeros.device,
                    reason_details=[
                        f"Cast zero tensor because it cannot be created directly from array with dtype {convert_input.dtype}"
                    ],
                )
                ConvertOp.build([zeros], [zero_output])
                zeros = zero_output

            CompareOp.build([convert_input, zeros], outputs, compare_direction="NE")
            return

        ConvertOp.build([convert_input], outputs)


@export.public_api(document_under="operations/functions")
@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
        "T2": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
    },
    dtype_constraints={"input": "T1", "dtype": "T2", constraints.RETURN_VALUE: "T2"},
    dtype_exceptions=[
        {"T1": "float8", "T2": "int4"},
        {"T1": "float8", "T2": "int8"},
        {"T1": "float8", "T2": "int64"},
        {"T1": "int4", "T2": "float8"},
        {"T1": "int4", "T2": "int8"},
        {"T1": "int4", "T2": "int64"},
    ],
)
def cast(input: "tripy.Tensor", dtype: "tripy.dtype") -> "tripy.Tensor":
    r"""
    Returns a tensor with the contents of the input tensor casted to the specified data type.

    For casts into quantized datatypes (:class:`int4` and :class:`float8`), this performs a per-tensor
    quantization into that datatype with scale 1.0; for casts `from` those datatypes, this performs
    a per-tensor dequantization with scale 1.0. Direct use of :func:`quantize` and :func:`dequantize` allows
    for finer control over these parameters.

    Args:
        input: The input tensor.
        dtype: The desired data type.

    Returns:
        A tensor containing the casted values.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.Tensor([1, 2], dtype=tp.int32)
        output = tp.cast(input, tp.float32)

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([1, 2], dtype=np.float32))

    .. seealso:: :func:`quantize`, :func:`dequantize`
    """
    from tripy.common.datatype import bool as tp_bool, int8, float32
    from tripy.frontend.trace.ops.dequantize import dequantize
    from tripy.frontend.trace.ops.quantize import quantize
    from tripy.frontend.trace.ops.utils import is_quantized_dtype

    if input.dtype == dtype:
        return input

    # Note: we check for int8 below because MLIR-TRT can handle it in ordinary conversions
    # even though it is a quantized dtype

    # If given a quantized input, dequantize before converting. If bool is the target dtype,
    # we do still need to quantize int8s because it compiles into an MLIR-TRT *comparison* op
    if is_quantized_dtype(input.dtype) and (input.dtype != int8 or dtype == tp_bool):
        dequant_dtype = float32
        input = dequantize(input, 1.0, dequant_dtype)
        if dtype == dequant_dtype:
            return input

    if is_quantized_dtype(dtype) and dtype != int8:
        if input.dtype != float32:
            input = Cast.build([input], float32)
        return quantize(input, 1.0, dtype)
    return Cast.build([input], dtype)
