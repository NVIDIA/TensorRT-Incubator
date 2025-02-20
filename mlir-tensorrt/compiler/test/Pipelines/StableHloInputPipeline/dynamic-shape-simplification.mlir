// RUN: mlir-tensorrt-opt %s -stablehlo-simplification-pipeline -verify-diagnostics

// This is a regression test that checks a proper diagnostic is raised when the simplification of dynamism exposes an
// error in the input IR that should be caught by verification routines of the static op variant.

func.func @main() -> tensor<?x?xf32> {
  %0 = stablehlo.constant dense<[2, 3]> : tensor<2xi32>
  %1 = stablehlo.dynamic_iota %0, dim = 0 : (tensor<2xi32>) -> tensor<?x?xf32>
  %2 = stablehlo.constant dense<[3, 2]> : tensor<2xi32>
  %3 = stablehlo.dynamic_iota %2, dim = 0 : (tensor<2xi32>) -> tensor<?x?xf32>
  %4 = stablehlo.constant dense<> : tensor<0xi32>
  %5 = stablehlo.get_dimension_size %1, dim = 0 : (tensor<?x?xf32>) -> tensor<i32>
  %6 = stablehlo.constant dense<1> : tensor<1xi32>
  %7 = stablehlo.dynamic_reshape %5, %6 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %8 = stablehlo.get_dimension_size %1, dim = 1 : (tensor<?x?xf32>) -> tensor<i32>
  %9 = stablehlo.constant dense<1> : tensor<1xi32>
  %10 = stablehlo.dynamic_reshape %8, %9 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %11 = stablehlo.concatenate %7, %10, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %12 = stablehlo.constant dense<0> : tensor<1xi32>
  %13 = stablehlo.constant dense<0> : tensor<1xi32>
  %14 = stablehlo.constant dense<1> : tensor<1xi32>
  %15 = stablehlo.real_dynamic_slice %11, %12, %13, %14 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %16 = stablehlo.concatenate %4, %15, dim = 0 : (tensor<0xi32>, tensor<0xi32>) -> tensor<0xi32>
  %17 = stablehlo.constant dense<> : tensor<0xi32>
  %18 = stablehlo.get_dimension_size %3, dim = 0 : (tensor<?x?xf32>) -> tensor<i32>
  %19 = stablehlo.constant dense<1> : tensor<1xi32>
  %20 = stablehlo.dynamic_reshape %18, %19 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %21 = stablehlo.get_dimension_size %3, dim = 1 : (tensor<?x?xf32>) -> tensor<i32>
  %22 = stablehlo.constant dense<1> : tensor<1xi32>
  %23 = stablehlo.dynamic_reshape %21, %22 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %24 = stablehlo.concatenate %20, %23, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %25 = stablehlo.constant dense<0> : tensor<1xi32>
  %26 = stablehlo.constant dense<0> : tensor<1xi32>
  %27 = stablehlo.constant dense<1> : tensor<1xi32>
  %28 = stablehlo.real_dynamic_slice %24, %25, %26, %27 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %29 = stablehlo.concatenate %17, %28, dim = 0 : (tensor<0xi32>, tensor<0xi32>) -> tensor<0xi32>
  %30 = stablehlo.maximum %16, %29 : tensor<0xi32>
  %31 = stablehlo.constant dense<0> : tensor<1xi32>
  // expected-error @below {{'stablehlo.slice' op inferred type(s) 'tensor<0xi32>' are incompatible with return type(s) of operation 'tensor<2xi32>'}}
  // expected-error @below {{'stablehlo.slice' op failed to infer returned types}}
  %32 = stablehlo.real_dynamic_slice %11, %13, %31, %14 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %33 = stablehlo.concatenate %30, %32, dim = 0 : (tensor<0xi32>, tensor<2xi32>) -> tensor<2xi32>
  %34 = stablehlo.dynamic_broadcast_in_dim %1, %33, dims = [0, 1] : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %35 = stablehlo.constant dense<0> : tensor<1xi32>
  %36 = stablehlo.real_dynamic_slice %24, %26, %35, %27 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %37 = stablehlo.concatenate %30, %36, dim = 0 : (tensor<0xi32>, tensor<2xi32>) -> tensor<2xi32>
  %38 = stablehlo.dynamic_broadcast_in_dim %3, %37, dims = [0, 1] : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %39 = stablehlo.dot_general %34, %38, contracting_dims = [1] x [0] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %39 : tensor<?x?xf32>
}
