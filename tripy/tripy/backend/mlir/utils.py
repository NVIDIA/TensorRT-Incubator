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

from tripy import utils
from tripy.common import datatype
from tripy.common.exception import OmitStackInfo, raise_error
from tripy.logging import logger


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


def get_max_upper_bounds():
    return sys.maxsize


def make_ir_context() -> ir.Context:
    ctx = MLIRContext()
    ctx.enable_multithreading(False)
    # Allow unregistered dialects to assign trt shape_profile attribute to stablehlo program.
    ctx.allow_unregistered_dialects = True
    return ctx


def get_mlir_dtype(dtype: "tripy.dtype"):
    """
    Converts a tripy data type to an MLIR data type.
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


def get_mlir_scalar_attr(dtype: "tripy.dtype", value):
    from tripy.common.datatype import floating

    # MLIR represents float dtypes as FloatAttr
    # and non-float dtypes as IntegerAttr
    attr_func = ir.FloatAttr.get if issubclass(dtype, floating) else ir.IntegerAttr.get
    return attr_func(get_mlir_dtype(dtype), value)


def list_to_dense_attr(data: List, dtype: "tripy.dtype"):
    if isinstance(data, numbers.Number):
        return [get_mlir_scalar_attr(dtype, data)]
    attrs = []
    for element in data:
        attrs.extend(list_to_dense_attr(element, dtype))
    return attrs


def get_mlir_quant_dtype(
    origin_dtype: "tripy.dtype",
    quant_dtype: "tripy.dtype",
    scale: float,
    zero_point: int,
    storage_type_min: int,
    storage_type_max: int,
):
    """
    Converts a tripy data type to an MLIR quantized data type.

    Args:
        origin_dtype: original data type to be quantized
        quant_dtype: target data type to quantize
        dtype: One of int4, int8, float8
        scale: scale value of quantized tensor
        zero_point: zero point of quantized tensor
        storage_type_min: min value of quantized dtype
        storage_type_max: max value of quantized dtype
    """
    from mlir_tensorrt.compiler.dialects import quant

    storage_type = get_mlir_dtype(quant_dtype)
    expressed_type = get_mlir_dtype(origin_dtype)
    return quant.UniformQuantizedType.get(
        quant.UniformQuantizedType.FLAG_SIGNED,
        storage_type,
        expressed_type,
        scale,
        zero_point,
        storage_type_min,
        storage_type_max,
    )


def make_mlir_tensor(
    dtype: "tripy.common.dtype", shape: Optional[Sequence[int]] = None, rank: Optional[int] = None
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


def remove_sym_attr(mlir_text: str) -> str:
    return re.sub(r"module @\S+ {", "module {", mlir_text)


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
    if dtype not in TRIPY_DTYPE_TO_MLIR_TRT:
        raise_error(f"Data type: '{dtype}' does not have a corresponding runtime data type")
    return TRIPY_DTYPE_TO_MLIR_TRT.get(dtype)


def convert_runtime_dtype_to_tripy_dtype(dtype: runtime.ScalarTypeCode) -> datatype.dtype:
    if dtype not in MLIR_TRT_TO_TRIPY_DTYPE:
        raise_error(f"Data type: '{dtype}' does not have a corresponding tripy data type")
    return MLIR_TRT_TO_TRIPY_DTYPE.get(dtype)


def is_any_dim_dynamic(mlir_tensor):
    """
    Returns true if any of the dimension in a mlir tensor is dynamic.
    """
    from mlir_tensorrt.compiler.dialects._ods_common import get_op_result_or_value

    type = get_op_result_or_value(mlir_tensor).type
    return any([type.is_dynamic_dim(i) for i in range(type.rank)])


class ShapeContext:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ShapeContext, cls).__new__(cls)
            from tripy.backend.mlir.compiler import Compiler

            cls._instance.compiler = Compiler(trt_builder_opt_level=0)
        return cls._instance

    def __init__(self):
        self.shape_caches = {}

    @classmethod
    def get_compiler(cls):
        return cls._instance.compiler

    @utils.log_time
    def get_shape_of_dynamic_trace_tensor(self, trace_tensor):
        from tripy.flat_ir.flat_ir import FlatIR
        from tripy.frontend.utils import topological_sort
        import copy

        def traverse_backwards(tensor, visited_tensors, visited_producers):
            """
            Recurse back from tensor to the inputs to the graph and store the visited tensors and nodes.
            """
            from tripy.frontend.trace.ops.unsqueeze import Unsqueeze

            if id(tensor) in visited_tensors:
                return

            visited_tensors[id(tensor)] = tensor
            if tensor.producer is not None:
                visited_producers[id(tensor.producer)] = tensor.producer
                # Special recursion conditions op by op basis.
                # Only recurse inputs which are used in output shape calculations.
                if isinstance(tensor.producer, Unsqueeze):
                    traverse_backwards(tensor.producer.inputs[1], visited_tensors, visited_producers)
                else:
                    # Naively recurse all the inputs until a constant or user input.
                    for input_tensor in tensor.producer.inputs:
                        traverse_backwards(input_tensor, visited_tensors, visited_producers)

        def find_inputs(graph_nodes):
            """
            Populates inputs of the topologically sorted graph.
            """
            id_graph_nodes = [id(n) for n in graph_nodes]
            return [t for op in graph_nodes for t in op.inputs if id(t.producer) not in id_graph_nodes]

        subgraph = FlatIR()
        visited_tensors = {}
        visited_producers = {}

        traverse_backwards(trace_tensor, visited_tensors, visited_producers)
        visited_producers = [v for _, v in visited_producers.items()]

        visited_producers = topological_sort(visited_producers)
        input_tensors = find_inputs(visited_producers)

        subgraph.inputs = [subgraph.register_tensor(inp.to_flat_ir()) for inp in input_tensors]
        subgraph.outputs = [subgraph.register_tensor(trace_tensor.to_flat_ir())]

        for op in visited_producers:
            inputs = [subgraph.register_tensor(inp.to_flat_ir()) for inp in op.inputs]
            outputs = [subgraph.register_tensor(out.to_flat_ir()) for out in op.outputs]
            op.to_flat_ir(copy.copy(inputs), copy.copy(outputs))
            subgraph.integrate_subgraph(inputs, outputs)

        mlir = subgraph.to_mlir()
        mlir_str = mlir.__str__()
        if mlir_str in self.shape_caches:
            return self.shape_caches[mlir_str]
        else:
            func_output_types = self.get_compiler().infer_shapes(mlir, subgraph)
            # Calculate the elapsed time
            assert len(func_output_types.results) == 1
            self.shape_caches[mlir_str] = func_output_types.results[0].shape
            return func_output_types.results[0].shape


def map_error_to_user_code_and_raise(flat_ir, exc, stderr):
    """
    Maps errors originating from the backend to user code and raises an error.
    This function must be called in the context of an active exception, as it may reraise
    the outer exception if the error does not originate from the backend.
    """
    from tripy.common.exception import TripyException

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
