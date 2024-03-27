from mlir_tensorrt.compiler import ir
from mlir_tensorrt.compiler.dialects import stablehlo, func as func_dialect
import pytest

# Assembly StableHLO code for element wise add
ASM = """
func.func @add_op_test_si4() {
  %0 = stablehlo.constant dense<[0, 1, 2, -3, 0]> : tensor<5xi4>
  %1 = stablehlo.constant dense<[-8, -1, 2, -3, 7]> : tensor<5xi4>
  %2 = stablehlo.add %0, %1 : tensor<5xi4>
  func.return
}
"""


class TestMLIRBindings:
    def make_ir_context(self) -> ir.Context:
        """Creates an MLIR context."""
        context = ir.Context()
        context.enable_multithreading(False)
        return context

    # Check if Python module for MLIR works correctly.
    def test_mlir_python_bindings(self):
        with self.make_ir_context() as context:
            m = ir.Module.parse(ASM)
            assert m is not None
            add_op = m.body.operations[0].regions[0].blocks[0].operations[2]
            assert add_op is not None and add_op.OPERATION_NAME == "stablehlo.add"
