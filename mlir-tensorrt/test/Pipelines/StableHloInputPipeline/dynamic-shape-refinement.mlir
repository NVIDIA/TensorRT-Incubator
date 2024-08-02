// RUN: mlir-tensorrt-opt %s -split-input-file -stablehlo-preprocessing-pipeline | FileCheck %s

// This test verifies that the StableHLO dynamic shape refinement and canonicalization passes
// are run in order to simplify dynamic op variants and unknown dimensions.

func.func @test_dynamic_refinement() -> tensor<?x?xf32> {
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<1xf32>
  %2 = stablehlo.constant dense<0.615870714> : tensor<1xf32>
  %3 = stablehlo.get_dimension_size %1, dim = 0 : (tensor<1xf32>) -> tensor<i32>
  %4 = stablehlo.reshape %3 : (tensor<i32>) -> tensor<1xi32>
  %5 = stablehlo.concatenate %4, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %6 = stablehlo.get_dimension_size %2, dim = 0 : (tensor<1xf32>) -> tensor<i32>
  %7 = stablehlo.reshape %6 : (tensor<i32>) -> tensor<1xi32>
  %8 = stablehlo.concatenate %7, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %9 = stablehlo.maximum %5, %8 : tensor<1xi32>
  %10 = stablehlo.dynamic_broadcast_in_dim %1, %9, dims = [0] : (tensor<1xf32>, tensor<1xi32>) -> tensor<1xf32>
  %11 = stablehlo.dynamic_broadcast_in_dim %2, %9, dims = [0] : (tensor<1xf32>, tensor<1xi32>) -> tensor<1xf32>
  %12 = stablehlo.compare  GE, %10, %11 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
  %13 = stablehlo.constant dense<[[1.51805913, 0.519577801, 1.0925405], [1.67694151, 1.300331, 8.165360e-01]]> : tensor<2x3xf32>
  %14 = stablehlo.constant dense<[[0.0593601689, 0.801895439, 0.449776143], [1.37200236, 0.118801221, 0.772216677]]> : tensor<2x3xf32>
  %15 = stablehlo.constant dense<1> : tensor<i32>
  %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<i32>) -> tensor<1xi32>
  %17 = stablehlo.get_dimension_size %12, dim = 0 : (tensor<1xi1>) -> tensor<i32>
  %18 = stablehlo.reshape %17 : (tensor<i32>) -> tensor<1xi32>
  %19 = stablehlo.concatenate %18, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %20 = stablehlo.concatenate %16, %19, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %21 = stablehlo.dynamic_broadcast_in_dim %12, %20, dims = [1] : (tensor<1xi1>, tensor<2xi32>) -> tensor<1x1xi1>
  %22 = stablehlo.get_dimension_size %13, dim = 0 : (tensor<2x3xf32>) -> tensor<i32>
  %23 = stablehlo.reshape %22 : (tensor<i32>) -> tensor<1xi32>
  %24 = stablehlo.get_dimension_size %13, dim = 1 : (tensor<2x3xf32>) -> tensor<i32>
  %25 = stablehlo.reshape %24 : (tensor<i32>) -> tensor<1xi32>
  %26 = stablehlo.concatenate %23, %25, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %27 = stablehlo.get_dimension_size %14, dim = 0 : (tensor<2x3xf32>) -> tensor<i32>
  %28 = stablehlo.reshape %27 : (tensor<i32>) -> tensor<1xi32>
  %29 = stablehlo.get_dimension_size %14, dim = 1 : (tensor<2x3xf32>) -> tensor<i32>
  %30 = stablehlo.reshape %29 : (tensor<i32>) -> tensor<1xi32>
  %31 = stablehlo.concatenate %28, %30, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %32 = stablehlo.maximum %26, %31 : tensor<2xi32>
  %33 = stablehlo.dynamic_broadcast_in_dim %21, %32, dims = [0, 1] : (tensor<1x1xi1>, tensor<2xi32>) -> tensor<?x?xi1>
  %34 = stablehlo.dynamic_broadcast_in_dim %13, %32, dims = [0, 1] : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %35 = stablehlo.dynamic_broadcast_in_dim %14, %32, dims = [0, 1] : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %36 = stablehlo.select %33, %34, %35 : tensor<?x?xi1>, tensor<?x?xf32>
  return %36 : tensor<?x?xf32>
}

// CHECK-LABEL: @test_dynamic_refinement
//  CHECK-SAME: () -> tensor<2x3xf32>
//  CHECK-NEXT:     %[[v0:.+]] = stablehlo.constant dense<{{\[}}[1.51805913, 0.519577801, 1.0925405], [1.67694151, 1.300331, 8.165360e-01]]> : tensor<2x3xf32>
//  CHECK-NEXT:     return %[[v0]] : tensor<2x3xf32>

// -----


func.func @test_refinement_dot_general() -> tensor<?x?xf32> {
  %0 = stablehlo.constant dense_resource<__elided__> : tensor<2x3xf32>
  %1 = stablehlo.constant dense_resource<__elided__> : tensor<3x2xf32>
  %2 = stablehlo.constant dense<> : tensor<0xi32>
  %3 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<2x3xf32>) -> tensor<i32>
  %4 = stablehlo.reshape %3 : (tensor<i32>) -> tensor<1xi32>
  %5 = stablehlo.get_dimension_size %0, dim = 1 : (tensor<2x3xf32>) -> tensor<i32>
  %6 = stablehlo.reshape %5 : (tensor<i32>) -> tensor<1xi32>
  %7 = stablehlo.concatenate %4, %6, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %8 = stablehlo.constant dense<0> : tensor<1xi32>
  %9 = stablehlo.constant dense<0> : tensor<1xi32>
  %10 = stablehlo.constant dense<1> : tensor<1xi32>
  %11 = stablehlo.real_dynamic_slice %7, %8, %9, %10 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %12 = stablehlo.concatenate %2, %11, dim = 0 : (tensor<0xi32>, tensor<0xi32>) -> tensor<0xi32>
  %13 = stablehlo.constant dense<> : tensor<0xi32>
  %14 = stablehlo.get_dimension_size %1, dim = 0 : (tensor<3x2xf32>) -> tensor<i32>
  %15 = stablehlo.reshape %14 : (tensor<i32>) -> tensor<1xi32>
  %16 = stablehlo.get_dimension_size %1, dim = 1 : (tensor<3x2xf32>) -> tensor<i32>
  %17 = stablehlo.reshape %16 : (tensor<i32>) -> tensor<1xi32>
  %18 = stablehlo.concatenate %15, %17, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %19 = stablehlo.constant dense<0> : tensor<1xi32>
  %20 = stablehlo.constant dense<0> : tensor<1xi32>
  %21 = stablehlo.constant dense<1> : tensor<1xi32>
  %22 = stablehlo.real_dynamic_slice %18, %19, %20, %21 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %23 = stablehlo.concatenate %13, %22, dim = 0 : (tensor<0xi32>, tensor<0xi32>) -> tensor<0xi32>
  %24 = stablehlo.maximum %12, %23 : tensor<0xi32>
  %25 = stablehlo.constant dense<2> : tensor<1xi32>
  %26 = stablehlo.real_dynamic_slice %7, %9, %25, %10 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %27 = stablehlo.concatenate %24, %26, dim = 0 : (tensor<0xi32>, tensor<2xi32>) -> tensor<2xi32>
  %28 = stablehlo.dynamic_broadcast_in_dim %0, %27, dims = [0, 1] : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %29 = stablehlo.constant dense<2> : tensor<1xi32>
  %30 = stablehlo.real_dynamic_slice %18, %20, %29, %21 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %31 = stablehlo.concatenate %24, %30, dim = 0 : (tensor<0xi32>, tensor<2xi32>) -> tensor<2xi32>
  %32 = stablehlo.dynamic_broadcast_in_dim %1, %31, dims = [0, 1] : (tensor<3x2xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %33 = stablehlo.dot_general %28, %32, contracting_dims = [1] x [0] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %33 : tensor<?x?xf32>
}

// CHECK-LABEL: @test_refinement_dot_general
//  CHECK-SAME: () -> tensor<2x2xf32>
//   CHECK-DAG:     %[[v0:.+]] = stablehlo.constant dense_resource<__elided__> : tensor<2x3xf32>
//   CHECK-DAG:     %[[v1:.+]] = stablehlo.constant dense_resource<__elided__> : tensor<3x2xf32>
//       CHECK:     %[[v2:.+]] = stablehlo.dot_general %[[v0]], %[[v1]], contracting_dims = [1] x [0] : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
//       CHECK:     return %[[v2]] : tensor<2x2xf32>

// -----

func.func @test_dynamic_gather_to_gather_refinement(%arg0: tensor<?xi32>) -> tensor<?x?xf32> {
  %0 = stablehlo.iota dim = 0 : tensor<4x6xf32>
  %1 = stablehlo.constant dense<1> : tensor<1xi32>
  %2 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<4x6xf32>) -> tensor<i32>
  %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
  %4 = stablehlo.get_dimension_size %0, dim = 1 : (tensor<4x6xf32>) -> tensor<i32>
  %5 = stablehlo.reshape %4 : (tensor<i32>) -> tensor<1xi32>
  %6 = stablehlo.concatenate %3, %5, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %7 = stablehlo.constant dense<1> : tensor<1xi32>
  %8 = stablehlo.constant dense<2> : tensor<1xi32>
  %9 = stablehlo.real_dynamic_slice %6, %7, %8, %1 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %10 = stablehlo.concatenate %1, %9, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %11 = "stablehlo.dynamic_gather"(%0, %arg0, %10) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1], collapsed_slice_dims = [0],
      start_index_map = [0], index_vector_dim = 1>
    } : (tensor<4x6xf32>, tensor<?xi32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %11 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @test_dynamic_gather_to_gather_refinement
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xi32>) -> tensor<?x6xf32> {
//       CHECK:     %[[v0:.+]] = stablehlo.iota dim = 0 : tensor<4xf32>
//       CHECK:     %[[v1:.+]] = stablehlo.broadcast_in_dim %[[v0]], dims = [0] : (tensor<4xf32>) -> tensor<4x6xf32>
//       CHECK:     %[[v2:.+]] = "stablehlo.gather"(%[[v1]], %[[arg0]]) {{.*}} : (tensor<4x6xf32>, tensor<?xi32>) -> tensor<?x6xf32>
//       CHECK:     return %[[v2]] : tensor<?x6xf32>

// -----

func.func @test_zero_size_slice_regression_test() -> tensor<?x?xf32> {
  %0 = stablehlo.constant dense<1> : tensor<1xi32>
  %1 = stablehlo.constant dense<[0.669720292, 0.609859764, 5.931760e-01, 0.613267303]> : tensor<4xf32>
  %2 = stablehlo.constant dense<4> : tensor<1xi32>
  %3 = stablehlo.slice %2 [1:1] : (tensor<1xi32>) -> tensor<?xi32>
  %4 = stablehlo.concatenate %2, %0, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<?xi32>) -> tensor<2xi32>
  %5 = stablehlo.dynamic_broadcast_in_dim %1, %4, dims = [0] : (tensor<4xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %5 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @test_zero_size_slice_regression_test
//  CHECK-NEXT:     %[[v0:.+]] = stablehlo.constant dense<{{\[}}[0.669720292], [0.609859764], [5.931760e-01], [0.613267303]]> : tensor<4x1xf32>
//  CHECK-NEXT:     return %[[v0]] : tensor<4x1xf32>


// -----

func.func @dynamic_slice_concat_regression_test() -> tensor<?xf32> {
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %1 = stablehlo.constant dense<4> : tensor<1xi32>
  %2 = stablehlo.dynamic_broadcast_in_dim %0, %1, dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
  %3 = stablehlo.constant dense<0> : tensor<1xi32>
  %4 = stablehlo.concatenate %3, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %5 = stablehlo.constant dense<1> : tensor<1xi32>
  %6 = stablehlo.concatenate %5, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %7 = stablehlo.constant dense<1> : tensor<1xi32>
  %8 = stablehlo.concatenate %7, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %9 = stablehlo.real_dynamic_slice %2, %4, %6, %8 : (tensor<?xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xf32>
  %10 = stablehlo.concatenate %5, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %11 = stablehlo.constant dense<2> : tensor<1xi32>
  %12 = stablehlo.concatenate %11, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %13 = stablehlo.constant dense<1> : tensor<1xi32>
  %14 = stablehlo.concatenate %13, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %15 = stablehlo.real_dynamic_slice %2, %10, %12, %14 : (tensor<?xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xf32>
  %16 = stablehlo.concatenate %11, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %17 = stablehlo.get_dimension_size %2, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %18 = stablehlo.constant dense<1> : tensor<1xi32>
  %19 = stablehlo.dynamic_reshape %17, %18 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %20 = stablehlo.concatenate %19, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %21 = stablehlo.constant dense<0> : tensor<1xi32>
  %22 = stablehlo.constant dense<1> : tensor<1xi32>
  %23 = stablehlo.constant dense<1> : tensor<1xi32>
  %24 = stablehlo.real_dynamic_slice %20, %21, %22, %23 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %25 = stablehlo.concatenate %24, dim = 0 : (tensor<?xi32>) -> tensor<1xi32>
  %26 = stablehlo.constant dense<1> : tensor<1xi32>
  %27 = stablehlo.concatenate %26, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %28 = stablehlo.real_dynamic_slice %2, %16, %25, %27 : (tensor<?xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xf32>
  return %28 : tensor<?xf32>
}

// CHECK-LABEL: func.func @dynamic_slice_concat_regression_test() -> tensor<2xf32>
//  CHECK-NEXT:   %[[cst:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
//  CHECK-NEXT:   return %[[cst]] : tensor<2xf32>
