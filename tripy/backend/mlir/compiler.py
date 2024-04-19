from typing import Optional, Tuple

import mlir_tensorrt.compiler.api as compiler
from mlir_tensorrt.compiler import ir

from tripy import config, utils
from tripy.backend.mlir.utils import parse_tensor_names_from_location, redirect_stderr, remove_constants
from tripy.common.exception import raise_error
import tripy.config as cfg
from tripy.logging import logger

G_MLIR_CONTEXT = None
G_COMPILER_CLIENT = None
G_TIMING_CACHE_FILE = cfg.timing_cache_file_path


def map_error_to_user_code_and_raise(flat_ir, exc, stderr):
    _, output_names, trace_input_names, trace_output_names, stderr = parse_tensor_names_from_location(stderr)

    assert (
        len(output_names) <= 1
    ), f"Error messages are only implemented for single output ops. Please fix if you see this message!"

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
                [f"\nNote: Tripy introduced new operation(s) in order to ", *out_tensor.reason_context, "."]
                if out_tensor.reason_context
                else []
            )
            + (
                [f"\nThis operation was introduced to ", *out_tensor.reason_details, "."]
                if out_tensor.reason_details
                else []
            )
            + [
                "\n\n",
            ]
        )

    raise_error(
        str(exc),
        details=[stderr, "\n"]
        + get_flat_ir_operation(output_names)
        + ["Note: This originated from the following expression:"]
        + get_tensors(trace_output_names)
        + get_tensors(trace_input_names, "Input"),
    )


# Avoid instantiating the compiler more than once.
def _get_compiler_objects() -> Tuple[ir.Context, compiler.CompilerClient]:
    global G_MLIR_CONTEXT, G_COMPILER_CLIENT, G_TIMING_CACHE_FILE
    if G_TIMING_CACHE_FILE != cfg.timing_cache_file_path:
        # Reinitialize the compiler if the timing cache file path has changed.
        global G_COMPILER_CLIENT
        G_COMPILER_CLIENT = None
        G_TIMING_CACHE_FILE = cfg.timing_cache_file_path

    if G_MLIR_CONTEXT is None or G_COMPILER_CLIENT is None:
        G_MLIR_CONTEXT = ir.Context()
        G_COMPILER_CLIENT = compiler.CompilerClient(G_MLIR_CONTEXT, compiler.CompilerClientOptions(G_TIMING_CACHE_FILE))
    return G_MLIR_CONTEXT, G_COMPILER_CLIENT


class Compiler:
    def __init__(self, trt_builder_opt_level=0) -> None:
        self.mlir_context, self.compiler_client = _get_compiler_objects()
        self.trt_builder_opt_level = trt_builder_opt_level

    def _make_mlir_opts(self, trt_builder_opt_level):
        opts = compiler.StableHLOToExecutableOptions(
            tensorrt_builder_opt_level=trt_builder_opt_level, tensorrt_strongly_typed=True
        )
        if config.enable_mlir_debug:
            opts.set_debug_options(config.enable_mlir_debug, config.mlir_debug_types, config.mlir_debug_tree_path)
        return opts

    def compile_stabehlo_program(self, code: str) -> compiler.Executable:
        with self.mlir_context:
            module = ir.Module.parse(code)
            opts = self._make_mlir_opts(self.trt_builder_opt_level)
            return compiler.compiler_stablehlo_to_executable(self.compiler_client, module.operation, opts)

    # The optional flat_ir parameter is used to generate nicer error messages.
    @utils.log_time
    def compile(self, mlir_module: ir.Module, flat_ir: Optional["FlatIR"] = None) -> compiler.Executable:
        logger.mlir(lambda: f"{utils.prefix_with_line_numbers(remove_constants(str(mlir_module)))}\n")
        opts = self._make_mlir_opts(self.trt_builder_opt_level)

        try:
            with redirect_stderr() as outfile:
                executable = compiler.compiler_stablehlo_to_executable(
                    self.compiler_client, mlir_module.operation, opts
                )
        except Exception as exc:
            outfile.flush()
            outfile.seek(0)
            stderr = outfile.read()

            map_error_to_user_code_and_raise(flat_ir, exc, stderr.decode())
        else:
            return executable
