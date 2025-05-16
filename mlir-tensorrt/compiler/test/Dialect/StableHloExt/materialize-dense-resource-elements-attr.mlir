// RUN: mlir-tensorrt-opt %s -split-input-file --stablehlo-ext-materialize-dense-resource-elements-attr | FileCheck %s

func.func @materialize_invalid() -> tensor<3xf32> {
    %c = stablehlo.constant dense<[10.0, 20.0, 30.0]> : tensor<3xf32>
    return %c : tensor<3xf32>
}

// CHECK-LABEL: @materialize_invalid
//       CHECK: stablehlo.constant dense<[{{.*}}]> : tensor<3xf32>

// -----

func.func @materialize_three_constants() -> (tensor<9xi32>, tensor<8xbf16>, tensor<8xi32>) {
    %c = stablehlo.constant dense<[10, 20, 30, 40, 50, 60, 70, 80, 90]> : tensor<9xi32>
    %c2 = stablehlo.constant dense<[10.38, 5.987, 24.18, 68.123, 98.99, 87.14, 47.65, 6.234]> : tensor<8xbf16>
    %c3 = stablehlo.constant dense<[10, 20, 30, 4, 50, 6, 70, 80]> : tensor<8xi32>
    return %c, %c2, %c3 : tensor<9xi32>, tensor<8xbf16>, tensor<8xi32>
}

// CHECK-LABEL: @materialize_three_constants
//  CHECK-SAME: () -> (tensor<9xi32>, tensor<8xbf16>, tensor<8xi32>)
//       CHECK: %[[c:.+]] = stablehlo.constant dense_resource<k_i32_1> : tensor<9xi32>
//       CHECK: %[[c2:.+]] = stablehlo.constant dense_resource<k_bf16> : tensor<8xbf16>
//       CHECK: %[[c3:.+]] = stablehlo.constant dense_resource<k_i32> : tensor<8xi32>
//       CHECK: return %[[c]], %[[c2]], %[[c3]] : tensor<9xi32>, tensor<8xbf16>, tensor<8xi32>
//       CHECK: {-#
//       CHECK:   dialect_resources: {
//       CHECK:     builtin: {
//   CHECK-DAG:       k_i32_1: "0x040000000A000000140000001E00000028000000320000003C00000046000000500000005A000000"
//   CHECK-DAG:       k_bf16: "0x020000002641C040C1418842C642AE423F42C740"
//   CHECK-DAG:       k_i32: "0x040000000A000000140000001E0000000400000032000000060000004600000050000000"
//       CHECK:     }
//       CHECK:   }
//       CHECK: #-}

// -----

func.func @materialize_one_constant_fp4() -> tensor<9xf4E2M1FN> {
    %c = stablehlo.constant dense<[-5.0, -4.1, -3.0, -2.2, -1.1, 1.8, 2.7, 3.0, 4.0]> : tensor<9xf4E2M1FN>
    return %c : tensor<9xf4E2M1FN>
}

// CHECK-LABEL: @materialize_one_constant_fp4
//  CHECK-SAME: () -> tensor<9xf4E2M1FN>
//       CHECK: %[[c:.+]] = stablehlo.constant dense_resource<k_f4E2M1FN> : tensor<9xf4E2M1FN>
//       CHECK: return %[[c]] : tensor<9xf4E2M1FN>
//       CHECK: {-#
//       CHECK:   dialect_resources: {
//       CHECK:     builtin: {
//       CHECK:       k_f4E2M1FN: "0x010000000E0E0D0C0A04050506"
//       CHECK:     }
//       CHECK:   }
//       CHECK: #-}