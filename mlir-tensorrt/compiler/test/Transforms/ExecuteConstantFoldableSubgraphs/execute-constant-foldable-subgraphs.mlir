// REQUIRES: cuda
// REQUIRES: host-has-at-least-1-gpus
// RUN: mlir-tensorrt-opt %s --plan-execute-constant-foldable-subgraphs --inline --canonicalize --split-input-file | FileCheck %s

func.func @test_i1() -> tensor<4xi1> {
  %0 = call @constant_subgraph() : () -> tensor<4xi1>
  return %0 : tensor<4xi1>
}
func.func private @constant_subgraph() -> tensor<4xi1> attributes {plan.constant_foldable} {
  %c = stablehlo.constant dense<[true, false, false, true]> : tensor<4xi1>
  %c_0 = stablehlo.constant dense<[true, true, true, false]> : tensor<4xi1>
  %0 = stablehlo.and %c, %c_0 : tensor<4xi1>
  return %0 : tensor<4xi1>
}

// CHECK-LABEL: @test_i1
//  CHECK-NEXT: %[[c:.+]] = arith.constant dense<[true, false, false, false]> : tensor<4xi1>
//  CHECK-NEXT:  return %[[c]] : tensor<4xi1>


// -----

func.func @test_i1_scalar() -> i1 {
  %0 = call @constant_subgraph() : () -> i1
  return %0 : i1
}
func.func private @constant_subgraph() -> i1 attributes {plan.constant_foldable} {
  %true = arith.constant true
  %false = arith.constant false
  %0 = arith.andi %true, %false : i1
  return %0 : i1
}

// CHECK-LABEL: @test_i1_scalar
//  CHECK-NEXT: %[[v0:.+]] = arith.constant false
//  CHECK-NEXT:  return %[[v0]] : i1

// -----

func.func @test_i8() -> tensor<4xi8> {
  %0 = call @constant_subgraph() : () -> tensor<4xi8>
  return %0 : tensor<4xi8>
}
func.func private @constant_subgraph() -> tensor<4xi8> attributes {plan.constant_foldable} {
  %c = stablehlo.constant dense<[10, 29, 50, 65]> : tensor<4xi8>
  %c_0 = stablehlo.constant dense<[90, 8, 5, 23]> : tensor<4xi8>
  %0 = stablehlo.add %c, %c_0 : tensor<4xi8>
  return %0 : tensor<4xi8>
}

// CHECK-LABEL: @test_i8
//  CHECK-NEXT: %[[c:.+]] = arith.constant dense<[100, 37, 55, 88]> : tensor<4xi8>
//  CHECK-NEXT: return %[[c]] : tensor<4xi8>

// -----

func.func @test_i8_scalar() -> i8 {
  %0 = call @constant_subgraph() : () -> i8
  return %0 : i8
}
func.func private @constant_subgraph() -> i8 attributes {plan.constant_foldable} {
  %c45_i8 = arith.constant 45 : i8
  %c30_i8 = arith.constant 30 : i8
  %0 = arith.addi %c45_i8, %c30_i8 : i8
  return %0 : i8
}

// CHECK-LABEL: @test_i8_scalar
//  CHECK-NEXT: %[[v0:.+]] = arith.constant 75
//  CHECK-NEXT:  return %[[v0]] : i8

// -----

func.func @test_i16() -> tensor<4xi16> {
  %0 = call @constant_subgraph() : () -> tensor<4xi16>
  return %0 : tensor<4xi16>
}
func.func private @constant_subgraph() -> tensor<4xi16> attributes {plan.constant_foldable} {
  %c = stablehlo.constant dense<[10, 29, 50, 65]> : tensor<4xi16>
  %c_0 = stablehlo.constant dense<[90, 8, 5, 23]> : tensor<4xi16>
  %0 = stablehlo.add %c, %c_0 : tensor<4xi16>
  return %0 : tensor<4xi16>
}

// CHECK-LABEL: @test_i16
//  CHECK-NEXT: %[[c:.+]] = arith.constant dense<[100, 37, 55, 88]> : tensor<4xi16>
//  CHECK-NEXT: return %[[c]] : tensor<4xi16>

// -----

func.func @test_i16_scalar() -> i16 {
  %0 = call @constant_subgraph() : () -> i16
  return %0 : i16
}
func.func private @constant_subgraph() -> i16 attributes {plan.constant_foldable} {
  %c125_i16 = arith.constant 125 : i16
  %c345_i16 = arith.constant 345 : i16
  %0 = arith.addi %c125_i16, %c345_i16 : i16
  return %0 : i16
}

// CHECK-LABEL: @test_i16_scalar
//  CHECK-NEXT: %[[v0:.+]] = arith.constant 470
//  CHECK-NEXT:  return %[[v0]] : i16

// -----

func.func @test_i32() -> tensor<4xi32> {
  %0 = call @constant_subgraph() : () -> tensor<4xi32>
  return %0 : tensor<4xi32>
}
func.func private @constant_subgraph() -> tensor<4xi32> attributes {plan.constant_foldable} {
  %c = stablehlo.constant dense<[10, 29, 507, 650]> : tensor<4xi32>
  %c_0 = stablehlo.constant dense<[980, 308, 765, 293]> : tensor<4xi32>
  %0 = stablehlo.add %c, %c_0 : tensor<4xi32>
  return %0 : tensor<4xi32>
}

// CHECK-LABEL: @test_i32
//  CHECK-NEXT: %[[c:.+]] = arith.constant dense<[990, 337, 1272, 943]> : tensor<4xi32>
//  CHECK-NEXT: return %[[c]] : tensor<4xi32>

// -----

func.func @test_i32_scalar() -> i32 {
  %0 = call @constant_subgraph() : () -> i32
  return %0 : i32
}
func.func private @constant_subgraph() -> i32 attributes {plan.constant_foldable} {
  %c1254_i32 = arith.constant 1254 : i32
  %c34545_i32 = arith.constant 34545 : i32
  %0 = arith.addi %c1254_i32, %c34545_i32 : i32
  return %0 : i32
}

// CHECK-LABEL: @test_i32_scalar
//  CHECK-NEXT: %[[v0:.+]] = arith.constant 35799
//  CHECK-NEXT:  return %[[v0]] : i32

// -----

func.func @test_i64() -> tensor<4xi64> {
  %0 = call @constant_subgraph() : () -> tensor<4xi64>
  return %0 : tensor<4xi64>
}
func.func private @constant_subgraph() -> tensor<4xi64> attributes {plan.constant_foldable} {
  %c = stablehlo.constant dense<[11230, 229, 250, 68875]> : tensor<4xi64>
  %c_0 = stablehlo.constant dense<[90, 12238, 29875, 23]> : tensor<4xi64>
  %0 = stablehlo.add %c, %c_0 : tensor<4xi64>
  return %0 : tensor<4xi64>
}

// CHECK-LABEL: @test_i64
//  CHECK-NEXT: %[[c:.+]] = arith.constant dense<[11320, 12467, 30125, 68898]> : tensor<4xi64>
//  CHECK-NEXT: return %[[c]] : tensor<4xi64>

// -----

func.func @test_i64_scalar() -> i64 {
  %0 = call @constant_subgraph() : () -> i64
  return %0 : i64
}
func.func private @constant_subgraph() -> i64 attributes {plan.constant_foldable} {
  %c12542_i64 = arith.constant 12542 : i64
  %c34545_i64 = arith.constant 34545 : i64
  %0 = arith.addi %c12542_i64, %c34545_i64 : i64
  return %0 : i64
}

// CHECK-LABEL: @test_i64_scalar
//  CHECK-NEXT: %[[v0:.+]] = arith.constant 47087
//  CHECK-NEXT:  return %[[v0]] : i64

// -----

func.func @test_f8_scalar() -> f8E4M3FN {
  %0 = call @constant_subgraph() : () -> f8E4M3FN
  return %0 : f8E4M3FN
}
func.func private @constant_subgraph() -> f8E4M3FN attributes {plan.constant_foldable} {
  %cst = arith.constant 2.250000e+00 : f8E4M3FN
  %cst_0 = arith.constant 4.062500e-01 : f8E4M3FN
  %0 = arith.addf %cst, %cst_0 : f8E4M3FN
  return %0 : f8E4M3FN
}

// CHECK-LABEL: @test_f8_scalar
//  CHECK-NEXT: %[[v0:.+]] = arith.constant 2.750000e+00
//  CHECK-NEXT:  return %[[v0]] : f8E4M3FN

// -----

func.func @test_f16() -> tensor<4xf16> {
  %0 = call @constant_subgraph() : () -> tensor<4xf16>
  return %0 : tensor<4xf16>
}
func.func private @constant_subgraph() -> tensor<4xf16> attributes {plan.constant_foldable} {
  %cst = stablehlo.constant dense<[3.820310e+00, 4.950000e+01, 1.899410e-01, 1.079690e+01]> : tensor<4xf16>
  %cst_0 = stablehlo.constant dense<[2.859380e+00, 2.023440e+01, 2.232500e+02, 3.310000e+02]> : tensor<4xf16>
  %0 = stablehlo.add %cst, %cst_0 : tensor<4xf16>
  return %0 : tensor<4xf16>
}

// CHECK-LABEL: @test_f16
//  CHECK-NEXT: %[[c:.+]] = arith.constant dense<[6.679690e+00, 6.975000e+01, 2.235000e+02, 3.417500e+02]> : tensor<4xf16>
//  CHECK-NEXT: return %[[c]] : tensor<4xf16>

// -----

func.func @test_f16_scalar() -> f16 {
  %0 = call @constant_subgraph() : () -> f16
  return %0 : f16
}
func.func private @constant_subgraph() -> f16 attributes {plan.constant_foldable} {
  %cst = arith.constant 2.925000e+01 : f16
  %cst_0 = arith.constant 6.250000e+01 : f16
  %0 = arith.addf %cst, %cst_0 : f16
  return %0 : f16
}

// CHECK-LABEL: @test_f16_scalar
//  CHECK-NEXT: %[[v0:.+]] = arith.constant 9.175000e+01
//  CHECK-NEXT:  return %[[v0]] : f16

// -----

func.func @test_f32() -> tensor<4xf32> {
  %0 = call @constant_subgraph() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
func.func private @constant_subgraph() -> tensor<4xf32> attributes {plan.constant_foldable} {
  %cst = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
  %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<4xf32>
  %0 = stablehlo.add %cst, %cst_0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: @test_f32
//  CHECK-NEXT: %[[c:.+]] = arith.constant dense<5.000000e+00> : tensor<4xf32>
//  CHECK-NEXT: return %[[c]] : tensor<4xf32>

// -----

func.func @test_f32_scalar() -> f32 {
  %0 = call @constant_subgraph() : () -> f32
  return %0 : f32
}
func.func private @constant_subgraph() -> f32 attributes {plan.constant_foldable} {
  %cst = arith.constant 2.092500e+02 : f32
  %cst_0 = arith.constant 6.325000e+02 : f32
  %0 = arith.addf %cst, %cst_0 : f32
  return %0 : f32
}

// CHECK-LABEL: @test_f32_scalar
//  CHECK-NEXT: %[[v0:.+]] = arith.constant 8.417500e+02
//  CHECK-NEXT:  return %[[v0]] : f32

// -----

func.func @test_f64() -> tensor<4xf64> {
  %0 = call @constant_subgraph() : () -> tensor<4xf64>
  return %0 : tensor<4xf64>
}
func.func private @constant_subgraph() -> tensor<4xf64> attributes {plan.constant_foldable} {
  %cst = stablehlo.constant dense<2.893000e+00> : tensor<4xf64>
  %cst_0 = stablehlo.constant dense<3.250000e+00> : tensor<4xf64>
  %0 = stablehlo.add %cst, %cst_0 : tensor<4xf64>
  return %0 : tensor<4xf64>
}

// CHECK-LABEL: @test_f64
//  CHECK-NEXT: %[[c:.+]] = arith.constant dense<6.143000e+00> : tensor<4xf64>
//  CHECK-NEXT: return %[[c]] : tensor<4xf64>

// -----

func.func @test_f64_scalar() -> f64 {
  %0 = call @constant_subgraph() : () -> f64
  return %0 : f64
}
func.func private @constant_subgraph() -> f64 attributes {plan.constant_foldable} {
  %cst = arith.constant 5.209250e+03 : f64
  %cst_0 = arith.constant 6.327550e+03 : f64
  %0 = arith.addf %cst, %cst_0 : f64
  return %0 : f64
}

// CHECK-LABEL: @test_f64_scalar
//  CHECK-NEXT: %[[v0:.+]] = arith.constant 1.153680e+04
//  CHECK-NEXT:  return %[[v0]] : f64

// -----

func.func @test_complex_f32() -> tensor<4xcomplex<f32>> {
  %0 = call @constant_subgraph() : () -> tensor<4xcomplex<f32>>
  return %0 : tensor<4xcomplex<f32>>
}
func.func private @constant_subgraph() -> tensor<4xcomplex<f32>> attributes {plan.constant_foldable} {
  %cst = stablehlo.constant dense<(1.0, 2.5)> : tensor<4xcomplex<f32>>
  %cst_0 = stablehlo.constant dense<(2.0, 3.9)> : tensor<4xcomplex<f32>>
  %0 = stablehlo.add %cst, %cst_0 : tensor<4xcomplex<f32>>
  return %0 : tensor<4xcomplex<f32>>
}

// CHECK-LABEL: @test_complex_f32
//  CHECK-NEXT: %[[c:.+]] = arith.constant dense<(3.000000e+00,6.400000e+00)> : tensor<4xcomplex<f32>>
//  CHECK-NEXT: return %[[c]] : tensor<4xcomplex<f32>>

// -----

func.func private @constant_subgraph() -> tensor<64xf16> attributes {plan.constant_foldable} {
    %cst_0 = stablehlo.constant dense_resource<__elided__> : tensor<64xf16>
    %cst_1 = stablehlo.constant dense_resource<__elided__> : tensor<64xf16>
    %r = stablehlo.add %cst_0, %cst_1 : tensor<64xf16>
    return %r : tensor<64xf16>
}

func.func @test_elided_f16() -> tensor<64xf16>{
    %0 = call @constant_subgraph() : () -> tensor<64xf16>
    return %0 : tensor<64xf16>
}

// CHECK-LABEL: @test_elided_f16
//  CHECK-NEXT: %[[cst:.+]] = arith.constant dense<2.000000e+00> : tensor<64xf16>
//  CHECK-NEXT: return %[[cst]] : tensor<64xf16>

// -----

func.func private @constant_subgraph() -> tensor<64xi8> attributes {plan.constant_foldable} {
    %cst_0 = stablehlo.constant dense_resource<__elided__> : tensor<64xi8>
    %cst_1 = stablehlo.constant dense_resource<__elided__> : tensor<64xi8>
    %r = stablehlo.add %cst_0, %cst_1 : tensor<64xi8>
    return %r : tensor<64xi8>
}

func.func @test_elided_int() -> tensor<64xi8>{
    %0 = call @constant_subgraph() : () -> tensor<64xi8>
    return %0 : tensor<64xi8>
}

// CHECK-LABEL: @test_elided_int
//  CHECK-NEXT: %[[cst:.+]] = arith.constant dense<2> : tensor<64xi8>
//  CHECK-NEXT: return %[[cst]] : tensor<64xi8>

// -----

func.func private @constant_subgraph() -> tensor<64xcomplex<f32>> attributes {plan.constant_foldable} {
    %cst_0 = stablehlo.constant dense_resource<__elided__> : tensor<64xcomplex<f32>>
    %cst_1 = stablehlo.constant dense_resource<__elided__> : tensor<64xcomplex<f32>>
    %r = stablehlo.add %cst_0, %cst_1 : tensor<64xcomplex<f32>>
    return %r : tensor<64xcomplex<f32>>
}

func.func @test_elided_complex() -> tensor<64xcomplex<f32>>{
    %0 = call @constant_subgraph() : () -> tensor<64xcomplex<f32>>
    return %0 : tensor<64xcomplex<f32>>
}

// CHECK-LABEL: @test_elided_complex
//  CHECK-NEXT: %[[cst:.+]] = arith.constant dense<(2.000000e+00,2.000000e+00)> : tensor<64xcomplex<f32>>
//  CHECK-NEXT: return %[[cst]] : tensor<64xcomplex<f32>>

// -----

func.func @test_complex_f32_scalar() -> complex<f32> {
  %0 = call @constant_subgraph() : () -> complex<f32>
  return %0 : complex<f32>
}
func.func private @constant_subgraph() -> complex<f32> attributes {plan.constant_foldable} {
  %cst0 = complex.constant [1.0 : f32, 2.0 : f32] : complex<f32>
  %cst1 = complex.constant [2.0 : f32, 3.0 : f32] : complex<f32>
  %0 = complex.add %cst0, %cst1: complex<f32>
  return %0 : complex<f32>
}

// CHECK-LABEL: @test_complex_f32_scalar
//  CHECK-NEXT: %[[c:.+]] = complex.constant [3.000000e+00 : f32, 5.000000e+00 : f32] : complex<f32>
//  CHECK-NEXT: return %[[c]] : complex<f32>

// -----

func.func @test_complex_f64_scalar() -> complex<f64> {
  %0 = call @constant_subgraph() : () -> complex<f64>
  return %0 : complex<f64>
}
func.func private @constant_subgraph() -> complex<f64> attributes {plan.constant_foldable} {
  %cst0 = complex.constant [20.4 : f64, 23.0 : f64] : complex<f64>
  %cst1 = complex.constant [23.5 : f64, 13.0 : f64] : complex<f64>
  %0 = complex.add %cst0, %cst1: complex<f64>
  return %0 : complex<f64>
}

// CHECK-LABEL: @test_complex_f64_scalar
//  CHECK-NEXT: %[[c:.+]] = complex.constant [4.390000e+01, 3.600000e+01] : complex<f64>
//  CHECK-NEXT: return %[[c]] : complex<f64>

// -----

func.func @test_multiple_results_no_alias() -> (tensor<i32>, tensor<1xi32>) {
  %0:2 = call @constant_subgraph() : () -> (tensor<i32>, tensor<1xi32>)
  return %0#0, %0#1 : tensor<i32>, tensor<1xi32>
}

func.func private @constant_subgraph() -> (tensor<i32>, tensor<1xi32>) attributes {plan.constant_foldable} {
  %c = stablehlo.constant dense<1018> : tensor<i32>
  %c_0 = stablehlo.constant dense<1> : tensor<i32>
  %c_1 = stablehlo.constant dense<0> : tensor<i32>
  %c_2 = stablehlo.constant dense<1> : tensor<i32>
  %0 = stablehlo.maximum %c, %c_2 : tensor<i32>
  %1 = stablehlo.compare  EQ, %0, %c_1,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %2 = stablehlo.select %1, %c_2, %0 : tensor<i1>, tensor<i32>
  %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
  return %2, %3 : tensor<i32>, tensor<1xi32>
}

// CHECK-LABEL: @test_multiple_results_no_alias
//  CHECK-NEXT: %[[c:.+]] = arith.constant dense<1018> : tensor<i32>
//  CHECK-NEXT: %[[c_0:.+]] = arith.constant dense<1018> : tensor<1xi32>
//  CHECK-NEXT: return %[[c]], %[[c_0]] : tensor<i32>, tensor<1xi32>
