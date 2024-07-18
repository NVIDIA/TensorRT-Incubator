import contextlib
import os
import re
import sys
import tempfile
from typing import BinaryIO, List, Tuple

import mlir_tensorrt.runtime.api as runtime
from mlir_tensorrt.compiler import ir

from tripy import utils
from tripy.common import ShapeInfo, datatype
from tripy.common.exception import raise_error
from tripy.logging import logger


def get_max_upper_bounds():
    return sys.maxsize


def make_ir_context() -> ir.Context:
    context = ir.Context()

    context.enable_multithreading(False)
    # Allow unregistered dialects to assign trt shape_profile attribute to stablehlo program.
    context.allow_unregistered_dialects = True
    return context


def get_mlir_dtype(dtype: "tripy.dtype"):
    """
    Converts a tripy data type to an MLIR data type.
    """
    return {
        "float64": ir.F64Type.get(),
        "float32": ir.F32Type.get(),
        "float16": ir.F16Type.get(),
        "float8": ir.Float8E4M3FNType.get(),
        "bfloat16": ir.BF16Type.get(),
        "int4": ir.IntegerType.get_signless(4),
        "int8": ir.IntegerType.get_signless(8),
        "int16": ir.IntegerType.get_signless(16),
        "int32": ir.IntegerType.get_signless(32),
        "int64": ir.IntegerType.get_signless(64),
        "uint8": ir.IntegerType.get_unsigned(8),
        "bool": ir.IntegerType.get_signless(1),
    }[dtype.name]


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


def make_mlir_tensor(shape, dtype: "tripy.common.dtype") -> ir.RankedTensorType:
    return ir.RankedTensorType.get(
        [ir.ShapedType.get_dynamic_size() if s.is_dynamic_dim() else s.min for s in utils.make_list(shape)],
        get_mlir_dtype(dtype),
    )


UNKNOWN_LOC = "unknown"
OUTPUT_SEPARATOR = ";;<out>;;"
TRACE_INPUTS_SEPARATOR = ";;<trace_in>;;"
TRACE_OUTPUTS_SEPARATOR = ";;<trace_out>;;"


def remove_sym_attr(mlir_text: str) -> str:
    return re.sub(r"module @\S+ {", "module {", mlir_text)


def remove_constants(mlir_text) -> str:
    lines = mlir_text.split("\n")

    def replace_dense_data(text):
        const_start_index = text.find("<") + 1
        const_end_index = text.find(">") - 1
        start_index = text.find(": tensor<") + 9

        substr = text[start_index:]
        dims = substr.split("x")
        dims = [int(dim) for dim in dims if dim.isdigit()]

        if utils.should_omit_constant_in_str(dims):
            return text[:const_start_index] + "..." + text[const_end_index + 1 :]
        return text

    replaced = [replace_dense_data(line) if "stablehlo.constant dense" in line else line for line in lines]
    return "\n".join(replaced)


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
TENSOR_NAME_PATTERN_NO_CAPTURE = re.compile(r'loc\("?.*?"?\):? ?')


def parse_tensor_names_from_location(msg: str) -> Tuple[List[str], List[str], str]:
    """
    Returns:
        The input names, output names, trace input names, trace output names, and new error message with location information stripped respectively.
    """
    locs = TENSOR_NAME_PATTERN.findall(msg)
    if not locs:
        return [], [], [], [], []

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
        return [], [], [], [], []

    input_names, _, loc = loc.partition(OUTPUT_SEPARATOR)
    output_names, _, loc = loc.partition(TRACE_INPUTS_SEPARATOR)
    trace_inputs, _, trace_outputs = loc.partition(TRACE_OUTPUTS_SEPARATOR)

    out_str = f"({trace_outputs})"

    return (
        input_names.split(","),
        output_names.split(","),
        trace_inputs.split(","),
        trace_outputs.split(","),
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
    datatype.int8: runtime.ScalarTypeCode.i8,
    datatype.int16: runtime.ScalarTypeCode.i16,
    datatype.int32: runtime.ScalarTypeCode.i32,
    datatype.int64: runtime.ScalarTypeCode.i64,
    datatype.uint8: runtime.ScalarTypeCode.ui8,
    datatype.float16: runtime.ScalarTypeCode.f16,
    datatype.float32: runtime.ScalarTypeCode.f32,
    datatype.float64: runtime.ScalarTypeCode.f64,
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
        raise_error(f"Data type: '{dtype}' does not have a corresponding numpy data type")
    return MLIR_TRT_TO_TRIPY_DTYPE.get(dtype)


def is_any_dim_dynamic(mlir_tensor):
    """
    Returns true if any of the dimension in a mlir tensor is dynamic.
    """
    assert hasattr(mlir_tensor, "type") or hasattr(mlir_tensor, "result")
    type = mlir_tensor.type if hasattr(mlir_tensor, "type") else mlir_tensor.result.type
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
        pass

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

        from tripy.common.device import device

        trace_tensor.device = device("cpu")
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
