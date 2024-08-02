# RUN: %PYTHON %s | FileCheck %s
from pathlib import Path

import mlir_tensorrt.compiler.api as api
import mlir_tensorrt.compiler.ir as ir

CANONICALIZER_STRESS_TEST_ASM = (
    Path(__file__).parent.parent.parent.parent
    / "Pipelines"
    / "StableHloInputPipeline"
    / "canonicalizer-stress-test.mlir"
).read_text()


def refine_signature(ASM):
    with ir.Context() as context:
        m = ir.Module.parse(ASM)
        client = api.CompilerClient(context)
        refined_func_type = api.get_stablehlo_program_refined_signature(
            client, m.operation, "main"
        )
        print(f"Refined func type: {refined_func_type}")


print("Testing StableHlo Program Signature Refinement")
refine_signature(CANONICALIZER_STRESS_TEST_ASM)
# CHECK-LABEL: Testing StableHlo Program Signature Refinement
# CHECK: Refined func type: () -> tensor<4xi32>
