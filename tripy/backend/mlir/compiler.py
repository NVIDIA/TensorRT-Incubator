from typing import Optional, Tuple

import mlir_tensorrt.compiler.api as compiler
from mlir_tensorrt.compiler import ir

import tripy.config as cfg
from tripy import config, utils
from tripy.backend.mlir.utils import (
    make_ir_context,
    map_error_to_user_code_and_raise,
    redirect_stderr,
    remove_constants,
)
from tripy.logging import logger

G_MLIR_CONTEXT = None
G_COMPILER_CLIENT = None
G_TIMING_CACHE_FILE = cfg.timing_cache_file_path


# Avoid instantiating the compiler more than once.
def _get_compiler_objects() -> Tuple[ir.Context, compiler.CompilerClient]:
    global G_MLIR_CONTEXT, G_COMPILER_CLIENT, G_TIMING_CACHE_FILE
    if G_TIMING_CACHE_FILE != cfg.timing_cache_file_path:
        # Reinitialize the compiler if the timing cache file path has changed.
        global G_COMPILER_CLIENT
        G_COMPILER_CLIENT = None
        G_TIMING_CACHE_FILE = cfg.timing_cache_file_path

    if G_MLIR_CONTEXT is None or G_COMPILER_CLIENT is None:
        G_MLIR_CONTEXT = make_ir_context()
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
        if config.enable_mlir_debug or config.enable_tensorrt_debug:
            opts.set_debug_options(
                config.enable_mlir_debug,
                config.mlir_debug_types if config.enable_mlir_debug else [],
                config.mlir_debug_tree_path if config.enable_mlir_debug else None,
                config.tensorrt_debug_path if config.enable_tensorrt_debug else None,
            )
        return opts

    def compile_stabehlo_program(self, code: str) -> compiler.Executable:
        with self.mlir_context:
            module = ir.Module.parse(code)
            opts = self._make_mlir_opts(self.trt_builder_opt_level)
            return compiler.compiler_stablehlo_to_executable(self.compiler_client, module.operation, opts)

    @utils.log_time
    def infer_shapes(self, mlir_module: ir.Module, flat_ir: Optional["FlatIR"] = None):
        try:
            with redirect_stderr() as outfile:
                refined_func_type = compiler.get_stablehlo_program_refined_signature(
                    self.compiler_client, mlir_module.operation, "main"
                )
        except Exception as exc:
            outfile.flush()
            outfile.seek(0)
            stderr = outfile.read()

            map_error_to_user_code_and_raise(flat_ir, exc, stderr.decode())
        else:
            return refined_func_type

    # The optional flat_ir parameter is used to generate nicer error messages.
    @utils.log_time
    def compile(self, mlir_module: ir.Module, flat_ir: Optional["FlatIR"] = None) -> compiler.Executable:
        logger.mlir(lambda: f"{remove_constants(str(mlir_module))}\n")
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
