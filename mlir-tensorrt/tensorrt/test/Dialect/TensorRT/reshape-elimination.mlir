// RUN: tensorrt-opt %s -split-input-file  -tensorrt-reshape-elimination | FileCheck %s

func.func @matmul_eliminate_reshape_lhs(%arg0: tensor<1x2x3x4xf16>, %arg1: tensor<4x2xf16>) -> tensor<1x2x3x2xf16>{
    %0 = tensorrt.reshape %arg0 : tensor<1x2x3x4xf16> to tensor<6x4xf16>
    %1 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>}
    ins(%0, %arg1 : tensor<6x4xf16>, tensor<4x2xf16>) -> tensor<6x2xf16>
    %2 = tensorrt.reshape %1 : tensor<6x2xf16> to tensor<1x2x3x2xf16>
    return %2: tensor<1x2x3x2xf16>
}

// CHECK-LABEL: @matmul_eliminate_reshape_lhs
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}})
//       CHECK: %[[v0:.+]] = tensorrt.expand_rank %[[arg1]]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.matrix_multiply {{{.*}}} ins(%[[arg0]], %[[v0]] : {{.*}})
//  CHECK-NEXT: return %[[v1]]

// -----

func.func @matmul_eliminate_reshape_lhs_2(%arg0: tensor<1x2x3x4x5x6xf16>, %arg1: tensor<1x2x6x8xf16>) -> tensor<1x2x3x4x5x8xf16>{
    %0 = tensorrt.reshape %arg0 : tensor<1x2x3x4x5x6xf16> to tensor<1x2x60x6xf16>
    %1 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>}
    ins(%0, %arg1 : tensor<1x2x60x6xf16>, tensor<1x2x6x8xf16>) -> tensor<1x2x60x8xf16>
    %2 = tensorrt.reshape %1 : tensor<1x2x60x8xf16> to tensor<1x2x3x4x5x8xf16>
    return %2: tensor<1x2x3x4x5x8xf16>
}

// CHECK-LABEL: @matmul_eliminate_reshape_lhs_2
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}})
//       CHECK: %[[v0:.+]] = tensorrt.expand_rank %[[arg1]]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.matrix_multiply {{{.*}}} ins(%[[arg0]], %[[v0]] : {{.*}})
//  CHECK-NEXT: return %[[v1]]

// -----

func.func @matmul_eliminate_reshape_lhs_3(%arg0: tensor<2x2x3x4xf16>, %arg1: tensor<2x4x5xf16>) -> tensor<2x2x3x5xf16>{
    %0 = tensorrt.reshape %arg0 : tensor<2x2x3x4xf16> to tensor<2x6x4xf16>
    %1 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>}
    ins(%0, %arg1 : tensor<2x6x4xf16>, tensor<2x4x5xf16>) -> tensor<2x6x5xf16>
    %2 = tensorrt.reshape %1 : tensor<2x6x5xf16> to tensor<2x2x3x5xf16>
    return %2: tensor<2x2x3x5xf16>
}

// CHECK-LABEL: @matmul_eliminate_reshape_lhs_3
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}})
//       CHECK: %[[v0:.+]] = tensorrt.expand_rank %[[arg1]]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.matrix_multiply {{{.*}}} ins(%[[arg0]], %[[v0]] : {{.*}})
//  CHECK-NEXT: return %[[v1]]
// -----

func.func @matmul_eliminate_reshape_lhs_negative(%arg0: tensor<10x20x30x40x50xf16>, %arg1: tensor<10x600x50x30xf16>) -> tensor<10x20x30x40x30xf16>{
    %0 = tensorrt.reshape %arg0 : tensor<10x20x30x40x50xf16> to tensor<10x600x40x50xf16>
    %1 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>}
    ins(%0, %arg1 : tensor<10x600x40x50xf16>, tensor<10x600x50x30xf16>) -> tensor<10x600x40x30xf16>
    %2 = tensorrt.reshape %1 : tensor<10x600x40x30xf16> to tensor<10x20x30x40x30xf16>
    return %2: tensor<10x20x30x40x30xf16>
}

// CHECK-LABEL: @matmul_eliminate_reshape_lhs_negative
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}})
//       CHECK: %[[v0:.+]] = tensorrt.reshape %[[arg0]]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.matrix_multiply {{{.*}}} ins(%[[v0]], %[[arg1]] : {{.*}})
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.reshape %[[v1]]
//  CHECK-NEXT: return %[[v2]]

// -----

func.func @matmul_eliminate_reshape_lhs_negative_dynamic(%arg0: tensor<10x?x30x40x50xf16>, %arg1: tensor<10x600x50x30xf16>) -> tensor<10x20x30x40x30xf16>{
    %0 = tensorrt.reshape %arg0 : tensor<10x?x30x40x50xf16> to tensor<10x600x40x50xf16>
    %1 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>}
    ins(%0, %arg1 : tensor<10x600x40x50xf16>, tensor<10x600x50x30xf16>) -> tensor<10x600x40x30xf16>
    %2 = tensorrt.reshape %1 : tensor<10x600x40x30xf16> to tensor<10x20x30x40x30xf16>
    return %2: tensor<10x20x30x40x30xf16>
}

// CHECK-LABEL: @matmul_eliminate_reshape_lhs_negative_dynamic
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}})
//       CHECK: %[[v0:.+]] = tensorrt.reshape %[[arg0]]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.matrix_multiply {{{.*}}} ins(%[[v0]], %[[arg1]] : {{.*}})
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.reshape %[[v1]]
//  CHECK-NEXT: return %[[v2]]

// -----

func.func @matmul_eliminate_reshape_lhs_negative_2(%arg0: tensor<1x2x3x4xf16>, %arg1: tensor<4x6xf16>) -> tensor<6x2x3xf16>{
    %0 = tensorrt.reshape %arg0 : tensor<1x2x3x4xf16> to tensor<6x4xf16>
    %1 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>}
    ins(%0, %arg1 : tensor<6x4xf16>, tensor<4x6xf16>) -> tensor<6x6xf16>
    %2 = tensorrt.reshape %1 : tensor<6x6xf16> to tensor<6x2x3xf16>
    return %2: tensor<6x2x3xf16>
}

// CHECK-LABEL: @matmul_eliminate_reshape_lhs_negative_2
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}})
//       CHECK: %[[v0:.+]] = tensorrt.reshape %[[arg0]]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.matrix_multiply {{{.*}}} ins(%[[v0]], %[[arg1]] : {{.*}})
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.reshape %[[v1]]
//  CHECK-NEXT: return %[[v2]]
// -----

func.func @matmul_simplify_reshape_rhs(%arg0: tensor<10x20x30x40xf16>, %arg1: tensor<200x60x30xf16>) -> tensor<10x20x60x40xf16>{
    %0 = tensorrt.reshape %arg0 : tensor<10x20x30x40xf16> to tensor<200x30x40xf16>
    %1 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>}
    ins(%arg1, %0 : tensor<200x60x30xf16>, tensor<200x30x40xf16>) -> tensor<200x60x40xf16>
    %2 = tensorrt.reshape %1 : tensor<200x60x40xf16> to tensor<10x20x60x40xf16>
    return %2: tensor<10x20x60x40xf16>
}

// CHECK-LABEL: @matmul_simplify_reshape_rhs
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}})
//       CHECK: %[[v0:.+]] = tensorrt.reshape %[[arg1]]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.matrix_multiply {{{.*}}} ins(%[[v0]], %[[arg0]] : {{.*}})
//  CHECK-NEXT: return %[[v1]]

// -----

func.func @matmul_simplify_reshape_rhs_2(%arg0: tensor<1x2x3x4x5x6xf16>, %arg1: tensor<1x2x12x6x5xf16>) -> tensor<1x2x3x4x6x6xf16>{
    %0 = tensorrt.reshape %arg0 : tensor<1x2x3x4x5x6xf16> to tensor<1x2x12x5x6xf16>
    %1 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>}
    ins(%arg1, %0 : tensor<1x2x12x6x5xf16>, tensor<1x2x12x5x6xf16>) -> tensor<1x2x12x6x6xf16>
    %2 = tensorrt.reshape %1 : tensor<1x2x12x6x6xf16> to tensor<1x2x3x4x6x6xf16>
    return %2: tensor<1x2x3x4x6x6xf16>
}

// CHECK-LABEL: @matmul_simplify_reshape_rhs_2
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}})
//       CHECK: %[[v0:.+]] = tensorrt.reshape %[[arg1]]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.matrix_multiply {{{.*}}} ins(%[[v0]], %[[arg0]] : {{.*}})
//  CHECK-NEXT: return %[[v1]]
// -----

func.func @matmul_simplify_reshape_rhs_negative(%arg0: tensor<1x2x3x4xf16>, %arg1: tensor<6x6xf16>) -> tensor<1x2x3x4xf16>{
    %0 = tensorrt.reshape %arg0 : tensor<1x2x3x4xf16> to tensor<6x4xf16>
    %1 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>}
    ins(%arg1, %0 : tensor<6x6xf16>, tensor<6x4xf16>) -> tensor<6x4xf16>
    %2 = tensorrt.reshape %1 : tensor<6x4xf16> to tensor<1x2x3x4xf16>
    return %2: tensor<1x2x3x4xf16>
}

// CHECK-LABEL: @matmul_simplify_reshape_rhs_negative
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}})
//       CHECK: %[[v0:.+]] = tensorrt.reshape %[[arg0]]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.matrix_multiply {{{.*}}} ins(%[[arg1]], %[[v0]] : {{.*}})
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.reshape %[[v1]]
//  CHECK-NEXT: return %[[v2]]

// -----

func.func @matmul_simplify_reshape_rhs_negative_dynamic(%arg0: tensor<?x?x?x4x5x6xf16>, %arg1: tensor<1x2x12x6x5xf16>) -> tensor<1x2x3x4x6x6xf16>{
    %0 = tensorrt.reshape %arg0 : tensor<?x?x?x4x5x6xf16> to tensor<1x2x12x5x6xf16>
    %1 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>}
    ins(%arg1, %0 : tensor<1x2x12x6x5xf16>, tensor<1x2x12x5x6xf16>) -> tensor<1x2x12x6x6xf16>
    %2 = tensorrt.reshape %1 : tensor<1x2x12x6x6xf16> to tensor<1x2x3x4x6x6xf16>
    return %2: tensor<1x2x3x4x6x6xf16>
}

// CHECK-LABEL: @matmul_simplify_reshape_rhs_negative_dynamic
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}})
//       CHECK: %[[v0:.+]] = tensorrt.reshape %[[arg0]]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.matrix_multiply {{{.*}}} ins(%[[arg1]], %[[v0]] : {{.*}})
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.reshape %[[v1]]
//  CHECK-NEXT: return %[[v2]]

// -----

func.func @sequential_reshape_elimination(%arg0: tensor<16x197x768xf16>) -> tensor<16x197x768xf16> {
    %0 = tensorrt.reshape %arg0 : tensor<16x197x768xf16> to tensor<3152x768xf16>
    %1 = tensorrt.reshape %0 : tensor<3152x768xf16> to tensor<16x197x768xf16>
    return %1 : tensor<16x197x768xf16>
}

// CHECK-LABEL: @sequential_reshape_elimination
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}})
//  CHECK-NEXT: return %[[arg0]]