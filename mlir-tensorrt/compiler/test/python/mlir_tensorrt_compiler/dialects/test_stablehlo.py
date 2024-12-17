# RUN: %PYTHON %s | FileCheck %s
from mlir_tensorrt.compiler.ir import *

ASM = """
func.func @test(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %0 = chlo.erf %arg0 : tensor<2x3x4xf32> -> tensor<2x3x4xf32>
  %1 = stablehlo.add %0, %0 : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  func.return %1 : tensor<2x3x4xf32>
}
"""

with Context() as context:
    m = Module.parse(ASM)
    assert m is not None
    block = m.body.operations[0].regions[0].blocks[0]
    assert block is not None
    assert block.operations[0] is not None
    assert block.operations[1] is not None
    assert block.operations[2] is not None
    print(block)


#      CHECK: ^bb0(%[[arg0:.+]]: tensor<2x3x4xf32>):
# CHECK-NEXT:   %[[v0:.+]] = chlo.erf %[[arg0]] : tensor<2x3x4xf32> -> tensor<2x3x4xf32>
# CHECK-NEXT:   %[[v1:.+]] = stablehlo.add %[[v0]], %[[v0]] : tensor<2x3x4xf32>
# CHECK-NEXT:   func.return %[[v1]] : tensor<2x3x4xf32>
