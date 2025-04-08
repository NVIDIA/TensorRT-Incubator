#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from nvtripy import export
from nvtripy.common.datatype import bool as tp_bool
from nvtripy.common.datatype import float32, int8
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.frontend.ops.dequantize import dequantize
from nvtripy.frontend.ops.quantize import quantize
from nvtripy.trace.ops.cast import Cast
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", "dtype": "T2", wrappers.RETURN_VALUE: "T2"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
        "T2": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
    },
    dtype_exceptions=[
        {"T1": "float8", "T2": "int4"},
        {"T1": "float8", "T2": "int8"},
        {"T1": "int8", "T2": "float8"},
        {"T1": "int4", "T2": "float8"},
        {"T1": "int4", "T2": "int8"},
        {"T1": "int4", "T2": "int64"},
    ],
)
def cast(input: "nvtripy.Tensor", dtype: "nvtripy.dtype") -> "nvtripy.Tensor":
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

        input = tp.Tensor([1, 2], dtype=tp.int32)
        output = tp.cast(input, tp.float32)

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([1, 2], dtype=np.float32))

    .. seealso:: :func:`quantize`, :func:`dequantize`
    """

    if input.dtype == dtype:
        return input

    # Note: we check for int8 below because MLIR-TRT can handle it in ordinary conversions
    # even though it is a quantized dtype

    # If given a quantized input, dequantize before converting. If bool is the target dtype,
    # we do still need to quantize int8s because it compiles into an MLIR-TRT *comparison* op
    if op_utils.is_quantized_dtype(input.dtype) and (input.dtype != int8 or dtype == tp_bool):
        dequant_dtype = float32
        input = dequantize(input, 1.0, dequant_dtype)
        if dtype == dequant_dtype:
            return input

    if op_utils.is_quantized_dtype(dtype) and dtype != int8:
        if input.dtype != float32:
            input = op_utils.create_op(Cast, [input], float32)
        return quantize(input, 1.0, dtype)
    return op_utils.create_op(Cast, [input], dtype)
