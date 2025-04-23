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

import contextlib
import numbers
import os
import re
import sys
import tempfile
import traceback
from typing import BinaryIO, List, Optional, Sequence, Tuple, Union

import mlir_tensorrt.runtime.api as runtime
from mlir_tensorrt.compiler import ir
from nvtripy import config, constants, utils
from nvtripy.common import datatype
from nvtripy.common.exception import raise_error
from nvtripy.logging import logger


# MLIR context needs to be shared across the Module op and CompilerClient
class MLIRContext:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.context = ir.Context()
        return cls._instance.context


# MLIR runtime needs to be initialized once.
class MLIRRuntimeClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.context = runtime.RuntimeClient()
        return cls._instance.context


def make_ir_context() -> ir.Context:
    ctx = MLIRContext()
    ctx.enable_multithreading(False)
    return ctx


def get_mlir_dtype(dtype: "nvtripy.dtype"):
    """
    Converts a nvtripy data type to an MLIR data type.
    """
    return {
        "float32": ir.F32Type.get(),
        "float16": ir.F16Type.get(),
        "float8": ir.Float8E4M3FNType.get(),
        "bfloat16": ir.BF16Type.get(),
        "int4": ir.IntegerType.get_signless(4),
        "int8": ir.IntegerType.get_signless(8),
        "int32": ir.IntegerType.get_signless(32),
        "int64": ir.IntegerType.get_signless(64),
        "bool": ir.IntegerType.get_signless(1),
    }[dtype.name]


def get_mlir_scalar_attr(mlir_dtype, value):
    # MLIR represents float dtypes as FloatAttr
    # and non-float dtypes as IntegerAttr
    attr_func = ir.IntegerAttr.get if isinstance(mlir_dtype, ir.IntegerType) else ir.FloatAttr.get
    return attr_func(mlir_dtype, value)


def list_to_dense_attr(data: List, mlir_dtype):
    from nvtripy.frontend.dimension_size import DimensionSize

    if isinstance(data, numbers.Number):
        return [get_mlir_scalar_attr(mlir_dtype, data)]

    if isinstance(data, DimensionSize):
        return [get_mlir_scalar_attr(mlir_dtype, data.tolist())]

    attrs = []
    for element in data:
        attrs.extend(list_to_dense_attr(element, mlir_dtype))
    return attrs


def make_mlir_tensor(
    dtype: "nvtripy.common.dtype",
    shape: Optional[Sequence[Union[int, "nvtripy.DimensionSize"]]] = None,
    rank: Optional[int] = None,
) -> ir.RankedTensorType:
    if shape is not None:
        return ir.RankedTensorType.get(
            [
                dim if isinstance(dim, int) and dim != constants.DYNAMIC_DIM else ir.ShapedType.get_dynamic_size()
                for dim in shape
            ],
            get_mlir_dtype(dtype),
        )

    assert rank is not None
    return ir.RankedTensorType.get(
        [ir.ShapedType.get_dynamic_size()] * rank,
        get_mlir_dtype(dtype),
    )


UNKNOWN_LOC = "unknown"
OUTPUT_SEPARATOR = ";;<out>;;"


def make_tensor_location(input_names: List[str], output_names: List[str]) -> ir.Location:
    return ir.Location.name(f"{','.join(input_names)}{OUTPUT_SEPARATOR}{','.join(output_names)}")


# The way locations are printed by MLIR-TRT differs from how they are printed by TRT, hence all the `?`s.
TENSOR_NAME_PATTERN = re.compile(r'loc\("?(.*?)"?\):? ?')
# Noncapturing pattern is required so that when we `.split`, we eliminate the entire pattern and not just
# the captured portions.
TENSOR_NAME_PATTERN_NO_CAPTURE = re.compile(r'loc\("?.*?"?\):? ?')


def parse_tensor_names_from_location(msg: str) -> Tuple[List[str], List[str], str]:
    """
    Returns:
        The input names, output names, and new error message with location information
        stripped respectively. If no location is found in the error message,
        the input/output names will all be empty lists.
    """
    locs = TENSOR_NAME_PATTERN.findall(msg)
    if not locs:
        return [], [], msg

    # TODO (#150): Update this logic to not only look at the first valid location attribute.
    loc = None
    contain_unknown_loc = False
    for l in locs:
        if not l or l == UNKNOWN_LOC:
            contain_unknown_loc = True
            continue
        else:
            loc = l
            break
    if contain_unknown_loc:
        logger.warning("Error location may be inaccurate as there are unknown locations from backend.")

    if not loc:
        return [], [], msg

    # Hack: Extract callsite for function call locations.
    AT_MARKER = 'at "'
    if AT_MARKER in loc:
        _, _, loc = loc.partition(AT_MARKER)
    input_names, _, output_names = loc.partition(OUTPUT_SEPARATOR)

    # Filter out empty names
    def remove_empty(lst):
        return list(filter(lambda x: x, lst))

    return (
        remove_empty(input_names.split(",")),
        remove_empty(output_names.split(",")),
        f"({output_names}) ".join(TENSOR_NAME_PATTERN_NO_CAPTURE.split(msg)),
    )


def map_error_to_user_code_and_raise(trace, exc, stderr):
    """
    Maps errors originating from the backend to user code and raises an error.
    This function must be called in the context of an active exception, as it may reraise
    the outer exception if the error does not originate from the backend.
    """
    from nvtripy.common.exception import TripyException

    # We don't want to do any additional processing for Tripy exceptions
    if isinstance(exc, TripyException):
        raise

    if hasattr(exc, "error_diagnostics"):
        stderr += ",".join(map(lambda err: str(err.location), exc.error_diagnostics))
    input_names, output_names, stderr = parse_tensor_names_from_location(stderr)

    def get_trace_operation():
        if trace is None or not output_names:
            return []

        inp_tensors = [trace.tensor_map[inp] for inp in input_names]
        out_tensors = [trace.tensor_map[out] for out in output_names]

        op = out_tensors[0].producer

        output_details = []
        if out_tensors:
            output_details = ["This originated from the following operation:", out_tensors[0]]
            if len(out_tensors) > 1:
                output_details = ["Note: Other outputs were: "] + out_tensors[1:]

        input_details = []
        if inp_tensors and "all" in config.extra_error_information:
            input_details = ["Inputs were:"] + inp_tensors

        return [
            *output_details,
            *input_details,
        ] + (
            [
                "This error occured while trying to compile the following Trace operation:",
                utils.utils.code_pretty_str(str(op)),
                "\n",
            ]
            if "all" in config.extra_error_information
            else []
        )

    # Construct the new exception with the formatted message
    error_message = f"{type(exc).__name__}: {str(exc)}"
    if "all" in config.extra_error_information:
        error_message += f"\n\nAdditional context:\n{traceback.format_exc()}"

    def starts_with_any(line, *starts):
        return any(line.strip().startswith(start) for start in starts)

    if "all" not in config.extra_error_information:
        # Strip out redundant error messages:
        new_stderr_lines = []
        first_error_found = False
        for line in stderr.splitlines():
            if starts_with_any(line, f"({','.join(output_names)}) error:", "error:"):
                if not first_error_found:
                    new_stderr_lines.append(line)
                first_error_found = True
            else:
                new_stderr_lines.append(line)
        stderr = "\n".join(new_stderr_lines)

    raise_error(
        error_message.replace("InternalError: InternalError:", "InternalError:").rstrip("."),
        details=[stderr, "\n\n"] + (get_trace_operation()),
    )


# For output originating outside Python, we need special logic to temporarily redirect the stderr
# file descriptor to something we can intercept. `contextlib.redirect_stderr` does not do this.
@contextlib.contextmanager
def redirect_stderr() -> BinaryIO:
    try:
        f = tempfile.NamedTemporaryFile()
        sys.stderr.flush()

        original_stderr = os.dup(2)
        new_stderr = os.dup(2)

        os.dup2(os.open(f.name, os.O_WRONLY | os.O_TRUNC | os.O_CREAT), 2)
        sys.stderr = os.fdopen(new_stderr, "w")

        yield f
    finally:
        sys.stderr.flush()

        os.dup2(original_stderr, 2)
        os.close(original_stderr)


TRIPY_DTYPE_TO_MLIR_TRT = {
    datatype.int4: runtime.ScalarTypeCode.i4,
    datatype.int8: runtime.ScalarTypeCode.i8,
    datatype.int32: runtime.ScalarTypeCode.i32,
    datatype.int64: runtime.ScalarTypeCode.i64,
    datatype.float16: runtime.ScalarTypeCode.f16,
    datatype.float32: runtime.ScalarTypeCode.f32,
    datatype.bool: runtime.ScalarTypeCode.i1,
    datatype.float8: runtime.ScalarTypeCode.f8e4m3fn,
    datatype.bfloat16: runtime.ScalarTypeCode.bf16,
}

MLIR_TRT_TO_TRIPY_DTYPE = {v: k for k, v in TRIPY_DTYPE_TO_MLIR_TRT.items()}


def convert_tripy_dtype_to_runtime_dtype(dtype: datatype.dtype) -> runtime.ScalarTypeCode:
    try:
        return TRIPY_DTYPE_TO_MLIR_TRT[dtype]
    except KeyError:
        raise_error(f"Data type: '{dtype}' does not have a corresponding runtime data type")


def convert_runtime_dtype_to_tripy_dtype(dtype: runtime.ScalarTypeCode) -> datatype.dtype:
    try:
        return MLIR_TRT_TO_TRIPY_DTYPE[dtype]
    except KeyError:
        raise_error(f"Data type: '{dtype}' does not have a corresponding nvtripy data type")


def is_any_dim_dynamic(mlir_tensor):
    """
    Returns true if any of the dimension in a mlir tensor is dynamic.
    """
    from mlir_tensorrt.compiler.dialects._ods_common import get_op_result_or_value

    type = get_op_result_or_value(mlir_tensor).type
    return any([type.is_dynamic_dim(i) for i in range(type.rank)])


def has_all_dynamic_dims(tensor_type: ir.RankedTensorType) -> bool:
    """Check if all dimensions of a tensor type are dynamic."""
    if not isinstance(tensor_type, ir.RankedTensorType):
        raise ValueError("Input must be a RankedTensorType")

    return all(dim == ir.ShapedType.get_dynamic_size() for dim in tensor_type.shape)
