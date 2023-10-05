# Import MLIR specific libraries from JAX
from jax._src.lib.mlir import ir
from jax._src.lib.mlir import dialects
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src.lib.mlir.dialects import hlo
import pytest

# Assembly MHLO code for element wise add
ASM = """
func.func @dynamicAdd(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = mhlo.add %arg0, %arg1 : tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}
"""


class TestMLIRFromJAX:
    def make_ir_context(self) -> ir.Context:
        """Creates an MLIR context suitable for JAX IR."""
        context = ir.Context()
        context.enable_multithreading(False)
        dialects.mhlo.register_mhlo_dialect(context)
        dialects.chlo.register_dialect(context)
        dialects.hlo.register_dialect(context)
        return context

    # Check if Python module for MLIR works correctly.
    def test_mlir_python_bindings_from_jax(self):
        with self.make_ir_context() as context:
            m = ir.Module.parse(ASM)
            assert m is not None
            add_op = m.body.operations[0].regions[0].blocks[0].operations[0]
            assert add_op is not None and add_op.OPERATION_NAME == "mhlo.add"
