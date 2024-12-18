// RUN: mlir-tensorrt-opt %s -convert-chlo-to-stablehlo-ext="preserve-erf=true preserve-topk=true" -split-input-file | FileCheck %s --check-prefix=CHECK
// RUN: mlir-tensorrt-opt %s -convert-chlo-to-stablehlo-ext="preserve-erf=false preserve-topk=false" -split-input-file | FileCheck %s --check-prefix=LOWERALL

func.func @erf_inv(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = chlo.erf_inv %arg0 : tensor<4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// We don't need to check the full lowering since that is tested upstream.
// We just test some basic logic and options.

// CHECK-LABEL: func.func @erf_inv
//   CHECK-NOT:  chlo.erf_inv

// LOWERALL-LABEL: func.func @erf_inv
//   LOWERALL-NOT:  chlo.erf_inv

// -----

func.func @erf(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = chlo.erf %arg0 : tensor<4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @erf
//  CHECK-NEXT:  chlo.erf

// LOWERALL-LABEL: func.func @erf
//   LOWERALL-NOT:  chlo.erf

// -----

func.func @top_k(%arg0: tensor<1x50257xf32>) -> (tensor<1x50xf32>, tensor<1x50xi32>) {
  %values, %indices = chlo.top_k(%arg0, k = 50) : tensor<1x50257xf32> -> (tensor<1x50xf32>, tensor<1x50xi32>)
  return %values, %indices : tensor<1x50xf32>, tensor<1x50xi32>
}

// CHECK-LABEL: func.func @top_k
//  CHECK-NEXT:  chlo.top_k

// LOWERALL-LABEL: func.func @top_k
//   LOWERALL-NOT:  chlo.top_k
