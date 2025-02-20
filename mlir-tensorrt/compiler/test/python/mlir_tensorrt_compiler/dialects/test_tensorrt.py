# RUN: %PYTHON %s | FileCheck %s

from mlir_tensorrt.compiler.dialects import tensorrt
from mlir_tensorrt.compiler.ir import *

ASM = """
func.func @test(%arg0: tensor<2x3x4xi32>) -> tensor<2x3x4xi32> {
  %0 = tensorrt.expand_rank %arg0 : tensor<2x3x4xi32> to tensor<1x2x3x4xi32>
  %1 = tensorrt.collapse_rank %0 : tensor<1x2x3x4xi32> to tensor<2x3x4xi32>
  return %1 : tensor<2x3x4xi32>
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


#      CHECK: ^bb0(%[[arg0:.+]]: tensor<2x3x4xi32>):
# CHECK-NEXT:  %[[v_0:.+]] = tensorrt.expand_rank %[[arg0]] : tensor<2x3x4xi32> to tensor<1x2x3x4xi32>
# CHECK-NEXT:  %[[v_1:.+]] = tensorrt.collapse_rank %0 : tensor<1x2x3x4xi32> to tensor<2x3x4xi32>
# CHECK-NEXT:  func.return %[[v_1]] : tensor<2x3x4xi32>


def test_attributes():
    with Context() as ctx:
        for attr in [
            tensorrt.ActivationTypeAttr.get("kRELU"),
            tensorrt.PaddingModeAttr.get("kEXPLICIT_ROUND_DOWN"),
            tensorrt.PoolingTypeAttr.get("kMAX"),
            tensorrt.ElementWiseOperationAttr.get("kSUM"),
            tensorrt.GatherModeAttr.get("kDEFAULT"),
            tensorrt.UnaryOperationAttr.get("kCOSH"),
            tensorrt.ReduceOperationAttr.get("kMAX"),
            tensorrt.SliceModeAttr.get("kCLAMP"),
            tensorrt.TopKOperationAttr.get("kMIN"),
            tensorrt.MatrixOperationAttr.get("kTRANSPOSE"),
            tensorrt.ResizeModeAttr.get("kLINEAR"),
            tensorrt.ResizeCoordinateTransformationAttr.get("kASYMMETRIC"),
            tensorrt.ResizeSelectorAttr.get("kFORMULA"),
            tensorrt.ResizeRoundModeAttr.get("kFLOOR"),
            tensorrt.LoopOutputAttr.get("kCONCATENATE"),
            tensorrt.TripLimitAttr.get("kWHILE"),
            tensorrt.FillOperationAttr.get("kRANDOM_UNIFORM"),
            tensorrt.ScatterModeAttr.get("kELEMENT"),
        ]:
            print(attr)


if __name__ == "__main__":
    test_attributes()

#      CHECK: #tensorrt.activation_type<kRELU>
# CHECK-NEXT: #tensorrt.padding_mode<kEXPLICIT_ROUND_DOWN>
# CHECK-NEXT: #tensorrt.pooling_type<kMAX>
# CHECK-NEXT: #tensorrt.element_wise_operation<kSUM>
# CHECK-NEXT: #tensorrt.gather_mode<kDEFAULT>
# CHECK-NEXT: #tensorrt.unary_operation<kCOSH>
# CHECK-NEXT: #tensorrt.reduce_operation<kMAX>
# CHECK-NEXT: #tensorrt.slice_mode<kCLAMP>
# CHECK-NEXT: #tensorrt.top_k_operation<kMIN>
# CHECK-NEXT: #tensorrt.matrix_operation<kTRANSPOSE>
# CHECK-NEXT: #tensorrt.resize_mode<kLINEAR>
# CHECK-NEXT: #tensorrt.resize_coordinate_transformation<kASYMMETRIC>
# CHECK-NEXT: #tensorrt.resize_selector<kFORMULA>
# CHECK-NEXT: #tensorrt.resize_round_mode<kFLOOR>
# CHECK-NEXT: #tensorrt.loop_output<kCONCATENATE>
# CHECK-NEXT: #tensorrt.trip_limit<kWHILE>
# CHECK-NEXT: #tensorrt.fill_operation<kRANDOM_UNIFORM>
# CHECK-NEXT: #tensorrt.scatter_mode<kELEMENT>
