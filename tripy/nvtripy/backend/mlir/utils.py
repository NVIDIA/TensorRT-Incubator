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

import contextlib
import numbers
import os
import re
import sys
import tempfile
from typing import BinaryIO, List, Tuple, Sequence, Optional
from itertools import chain
import traceback

import mlir_tensorrt.runtime.api as runtime
from mlir_tensorrt.compiler import ir

from nvtripy import utils
from nvtripy.common import datatype
from nvtripy.common.exception import OmitStackInfo, raise_error
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
    # Allow unregistered dialects to assign trt shape_profile attribute to stablehlo program.
    ctx.allow_unregistered_dialects = True
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
    dtype: "nvtripy.common.dtype", shape: Optional[Sequence[int]] = None, rank: Optional[int] = None
) -> ir.RankedTensorType:
    if shape is not None:
        return ir.RankedTensorType.get(
            [ir.ShapedType.get_dynamic_size() if dim < 0 else dim for dim in shape], get_mlir_dtype(dtype)
        )

    assert rank is not None
    return ir.RankedTensorType.get(
        [ir.ShapedType.get_dynamic_size()] * rank,
        get_mlir_dtype(dtype),
    )


def get_constant_value(arg) -> Optional[ir.DenseElementsAttr]:
    from mlir_tensorrt.compiler.dialects import stablehlo

    if isinstance(arg, ir.Value) and ir.OpResult.isinstance(arg):
        arg = ir.OpResult(arg).owner

    if isinstance(arg, ir.Operation):
        arg = arg.opview

    if isinstance(arg, stablehlo.ConstantOp):
        return arg.value

    return None


def check_tensor_type_and_suggest_contiguous(obj):
    obj_type = str(type(obj))
    if "torch.Tensor" in obj_type:
        return "PyTorch Tensor", "tensor.contiguous() or tensor.clone()"
    elif "jaxlib" in obj_type or "jax.numpy" in obj_type:
        return "JAX Array", "jax.numpy.asarray(array) or jax.numpy.copy(array)"
    elif "numpy.ndarray" in obj_type:
        return "NumPy Array", "np.ascontiguousarray(array) or array.copy(order='C')"
    elif "cupy.ndarray" in obj_type:
        return "CuPy Array", "cp.ascontiguousarray(array) or array.copy(order='C')"
    else:
        return None, None


UNKNOWN_LOC = "unknown"
OUTPUT_SEPARATOR = ";;<out>;;"
TRACE_INPUTS_SEPARATOR = ";;<trace_in>;;"
TRACE_OUTPUTS_SEPARATOR = ";;<trace_out>;;"


def make_tensor_location(
    input_names: List[str], output_names: List[str], trace_input_names: List[str], trace_output_names: List[str]
) -> ir.Location:
    return ir.Location.name(
        f"{','.join(input_names)}"
        f"{OUTPUT_SEPARATOR}{','.join(output_names)}"
        f"{TRACE_INPUTS_SEPARATOR}{','.join(trace_input_names)}"
        f"{TRACE_OUTPUTS_SEPARATOR}{','.join(trace_output_names)}"
    )


# The way locations are printed by MLIR-TRT differs from how they are printed by TRT, hence all the `?`s.
TENSOR_NAME_PATTERN = re.compile(r'loc\("?(.*?)"?\):? ?')
# Noncapturing pattern is required so that when we `.split`, we eliminate the entire pattern and not just
# the captured portions.
TENSOR_NAME_PATTERN_NO_CAPTURE = re.compile(r'loc\("?.*?"?\):? ?')


def parse_tensor_names_from_location(msg: str) -> Tuple[List[str], List[str], List[str], List[str], str]:
    """
    Returns:
        The input names, output names, trace input names, trace output names,
        and new error message with location information stripped respectively.
        If no location is found in the error message, the input/output names will all
        be empty lists.
    """
    locs = TENSOR_NAME_PATTERN.findall(msg)
    if not locs:
        return [], [], [], [], msg

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
        return [], [], [], [], msg

    # Hack: Extract callsite for function call locations.
    if "at" in loc:
        _, _, loc = loc.partition('at "')
    input_names, _, loc = loc.partition(OUTPUT_SEPARATOR)
    output_names, _, loc = loc.partition(TRACE_INPUTS_SEPARATOR)
    trace_inputs, _, trace_outputs = loc.partition(TRACE_OUTPUTS_SEPARATOR)

    out_str = f"({trace_outputs})"

    # Filter out empty names
    def remove_empty(lst):
        return list(filter(lambda x: x, lst))

    return (
        remove_empty(input_names.split(",")),
        remove_empty(output_names.split(",")),
        remove_empty(trace_inputs.split(",")),
        remove_empty(trace_outputs.split(",")),
        out_str.join(TENSOR_NAME_PATTERN_NO_CAPTURE.split(msg)),
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


def cast_to_dynamic_ranked_tensor(input_tensor: ir.Value, always_insert_cast: bool = False) -> ir.Value:
    """Cast a tensor to a dynamic ranked tensor if necessary."""
    from mlir_tensorrt.compiler.dialects._ods_common import get_op_result_or_value
    from mlir_tensorrt.compiler.dialects import stablehlo

    input_type = get_op_result_or_value(input_tensor).type

    if not ir.RankedTensorType.isinstance(input_type):
        raise ValueError("Input must be a RankedTensorType")

    if not always_insert_cast and has_all_dynamic_dims(input_type):
        return input_tensor

    dynamic_shape = [ir.ShapedType.get_dynamic_size()] * input_type.rank
    dynamic_type = ir.RankedTensorType.get(dynamic_shape, input_type.element_type)

    return stablehlo.ConvertOp(result=dynamic_type, operand=input_tensor).result


def map_error_to_user_code_and_raise(flat_ir, exc, stderr):
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
    _, output_names, trace_input_names, trace_output_names, stderr = parse_tensor_names_from_location(stderr)

    assert (
        len(output_names) <= 1
    ), f"Error messages are only implemented for single output ops. Please fix if you see this message!"

    def omit_stack_info(details):
        return list(map(lambda x: OmitStackInfo(x), details))

    def get_tensors(names, title=None):
        infos = []
        if flat_ir is None:
            return infos

        if not names or any(name not in flat_ir.tensor_map for name in names):
            return infos

        for index, name in enumerate(names):
            if title:
                infos.append(f"{title} {index}:")
            infos.append(flat_ir.tensor_map[name])
        return infos

    def interleave_newline(arr):
        return list(chain(*[omit_stack_info(sublist) + ["\n"] for sublist in arr]))[:-1]

    def get_flat_ir_operation(output_names):
        assert len(output_names) <= 1, f"Only implemented for single output ops"
        if not output_names or flat_ir is None:
            return []

        output_name = output_names[0]
        out_tensor = flat_ir.tensor_map[output_name]

        if output_name not in trace_output_names:
            # TODO (#165): Enforce reason_context like we do reason_details?
            assert (
                out_tensor.reason_details
            ), f"All intermediate tensors should have reason_details set, but {out_tensor} does not!"

        op = out_tensor.producer

        return (
            [
                "This error occured while trying to compile the following FlatIR expression:",
                utils.code_pretty_str(str(op)),
                "\n",
            ]
            + (
                [
                    f"\nNote: Tripy introduced new operation(s) in order to ",
                    *interleave_newline(out_tensor.reason_context),
                    ".",
                ]
                if out_tensor.reason_context
                else []
            )
            + (
                [
                    f"\nThis operation was introduced to ",
                    *omit_stack_info(out_tensor.reason_details),
                    ".",
                ]
                if out_tensor.reason_details
                else []
            )
            + [
                "\n\n",
            ]
        )

    # Construct the new exception with the formatted message
    error_message = f"{type(exc).__name__}: {str(exc)}\n\nAdditional context:\n{traceback.format_exc()}"

    raise_error(
        error_message.replace("InternalError: InternalError:", "InternalError:").rstrip(".") + ".",
        details=[stderr, "\n"]
        + (get_flat_ir_operation(output_names) if output_names else [])
        + (
            (
                ["Note: This originated from the following expression:"]
                + get_tensors(trace_output_names)
                + get_tensors(trace_input_names, "Input")
            )
            if trace_output_names or trace_input_names
            else []
        ),
    )
