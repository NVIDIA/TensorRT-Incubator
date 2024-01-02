from tripy.backend.mlir.mlir import mlir_wrapper, void_ptr
from tripy.backend.mlir.mlir_translator import lower_flat_ir_to_mlir
from tripy.common.logging import G_LOGGER
from tripy.util import log_time
from tripy.util.util import prefix_with_line_numbers


class FlatIRCompiler:
    """
    Represents the compiler for Trace, which converts Trace into a StableHLO representation
    and compiles it into an executable using mlir-tensorrt compiler.
    """

    def __init__(self) -> None:
        self.compiler = mlir_wrapper()

    @log_time
    def compile(self, flat_ir: "FlatIR") -> void_ptr:
        """
        Given a flat_ir, compile function traces the computation graph and generates a mlir-tensorrt executable.

        Returns:
            Pointer to the executable generated by the mlir-tensorrt compiler.
        """
        # Lower Trace to corresponding StableHLO IR.
        mlir_module = lower_flat_ir_to_mlir(flat_ir)
        mlir_textual = mlir_module.__str__()
        G_LOGGER.ir_printer(f"StableHLO IR:\n{prefix_with_line_numbers(mlir_textual)}")
        return self.compiler.compile(mlir_textual)
