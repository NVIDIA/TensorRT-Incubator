from tripy import utils
from tripy.backend.mlir.mlir import mlir_wrapper, void_ptr
from tripy.common.logging import logger


class FlatIRCompiler:
    """
    Represents the compiler for Trace, which converts Trace into a StableHLO representation
    and compiles it into an executable using mlir-tensorrt compiler.
    """

    def __init__(self) -> None:
        self.compiler = mlir_wrapper()

    @staticmethod
    def remove_stablehlo_constants(mlir_textual) -> str:
        lines = mlir_textual.split("\n")

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

    @utils.log_time
    def compile(self, flat_ir: "FlatIR") -> void_ptr:
        """
        Given a flat_ir, compile function traces the computation graph and generates a mlir-tensorrt executable.

        Returns:
            Pointer to the executable generated by the mlir-tensorrt compiler.
        """
        # Lower Trace to corresponding StableHLO IR.
        mlir_module = flat_ir.to_mlir()
        mlir_textual = str(mlir_module)
        logger.stablehlo(
            lambda: f"{utils.prefix_with_line_numbers(FlatIRCompiler.remove_stablehlo_constants(mlir_textual))}"
        )
        return self.compiler.compile(mlir_textual)
