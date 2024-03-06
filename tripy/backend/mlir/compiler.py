from typing import Tuple

import mlir_tensorrt.compiler.api as compiler
from mlir_tensorrt.compiler import ir

from tripy import utils
from tripy.common.logging import logger
from tripy.backend.mlir.utils import remove_constants

G_MLIR_CONTEXT = None
G_COMPILER_CLIENT = None


# Avoid instantiating the compiler more than once.
def _get_compiler_objects() -> Tuple[ir.Context, compiler.CompilerClient]:
    global G_MLIR_CONTEXT, G_COMPILER_CLIENT
    if G_MLIR_CONTEXT is None or G_COMPILER_CLIENT is None:
        G_MLIR_CONTEXT = ir.Context()
        G_COMPILER_CLIENT = compiler.CompilerClient(G_MLIR_CONTEXT, compiler.CompilerClientOptions())
    return G_MLIR_CONTEXT, G_COMPILER_CLIENT


class Compiler:
    def __init__(self) -> None:
        self.mlir_context, self.compiler_client = _get_compiler_objects()

    def compile_stabehlo_program(self, code: str) -> compiler.Executable:
        with self.mlir_context:
            module = ir.Module.parse(code)
            opts = compiler.StableHLOToExecutableOptions(tensorrt_builder_opt_level=0, tensorrt_strongly_typed=True)
            return compiler.compiler_stablehlo_to_executable(self.compiler_client, module.operation, opts)

    @utils.log_time
    def compile(self, mlir_module: ir.Module) -> compiler.Executable:
        logger.mlir(lambda: f"{utils.prefix_with_line_numbers(remove_constants(str(mlir_module)))}\n")
        opts = compiler.StableHLOToExecutableOptions(tensorrt_builder_opt_level=0, tensorrt_strongly_typed=True)
        return compiler.compiler_stablehlo_to_executable(self.compiler_client, mlir_module.operation, opts)
