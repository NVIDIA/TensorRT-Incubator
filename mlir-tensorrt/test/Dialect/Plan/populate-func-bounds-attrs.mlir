// RUN: mlir-tensorrt-opt %s -split-input-file -plan-populate-func-bounds-attrs | FileCheck %s

func.func public @single_return(%arg0: tensor<?xi32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [1], opt = [2], max = [3]>},
                                %arg1: tensor<?xi32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [1], opt = [2], max = [3]>}) -> tensor<?xi32> {
  %c0 = arith.constant 0 : index
  %0 = stablehlo.add %arg0, %arg1 : tensor<?xi32>
  %dim = tensor.dim %arg0, %c0 : tensor<?xi32>
  %1 = plan.with_shape %0(%dim) : (tensor<?xi32>, index) -> tensor<?xi32>
  return %1 : tensor<?xi32>
}

// CHECK-LABEL: @single_return
// CHECK-SAME: -> (tensor<?xi32> {tensorrt.shape_profile = #plan.bounds<shape, [1], [3]>})

// -----

func.func public @multiple_return(%arg0: tensor<?xi32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [1], opt = [2], max = [3]>}, %arg1: tensor<?xi32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [2], opt = [4], max = [6]>}) -> (tensor<?xi32>, tensor<?xi32>) {
  %c0 = arith.constant 0 : index
  %0 = stablehlo.add %arg0, %arg0 : tensor<?xi32>
  %1 = stablehlo.add %arg1, %arg1 : tensor<?xi32>
  %dim_1 = tensor.dim %arg0, %c0 : tensor<?xi32>
  %dim_2 = tensor.dim %arg1, %c0 : tensor<?xi32>
  %2 = plan.with_shape %0(%dim_1) : (tensor<?xi32>, index) -> tensor<?xi32>
  %3 = plan.with_shape %1(%dim_2) : (tensor<?xi32>, index) -> tensor<?xi32>
  return %2, %3 : tensor<?xi32>, tensor<?xi32>
}

// CHECK-LABEL: @multiple_return
// CHECK-SAME: -> (tensor<?xi32> {tensorrt.shape_profile = #plan.bounds<shape, [1], [3]>}, tensor<?xi32> {tensorrt.shape_profile = #plan.bounds<shape, [2], [6]>})

// -----

func.func public @scalar_return(%arg0: tensor<?xi32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [1], opt = [2], max = [3]>}) -> i32 {
  %c0 = arith.constant 0 : index
  %0 = stablehlo.add %arg0, %arg0 : tensor<?xi32>
  %1 = tensor.extract %0[%c0] : tensor<?xi32>
  return %1 : i32
}

// CHECK-LABEL: @scalar_return
// CHECK-SAME: -> i32

// -----

func.func public @static_return(%arg0: tensor<?xi32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [1], opt = [2], max = [3]>}) -> tensor<1xi32> {
  %c0 = arith.constant 0 : index
  %0 = stablehlo.add %arg0, %arg0 : tensor<?xi32>
  %1 = tensor.extract %0[%c0] : tensor<?xi32>
  %2 = tensor.from_elements %1 : tensor<1xi32>
  return %2 : tensor<1xi32>
}

// CHECK-LABEL: @static_return
// CHECK-SAME: -> tensor<1xi32> {

// -----

func.func @mixed_dims(%arg0: tensor<?x10xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [1, 10], opt = [2, 10], max = [3, 10]>}) -> tensor<?x10xf32> {
  %c10 = arith.constant 10 : index
  %c0 = arith.constant 0 : index
  %0 = stablehlo.exponential %arg0 : tensor<?x10xf32>
  %dim = tensor.dim %arg0, %c0 : tensor<?x10xf32>
  %1 = plan.with_shape %0(%dim, %c10) : (tensor<?x10xf32>, index, index) -> tensor<?x10xf32>
  return %1 : tensor<?x10xf32>
}

// CHECK-LABEL: @mixed_dims
// CHECK-SAME: -> (tensor<?x10xf32> {tensorrt.shape_profile = #plan.bounds<shape, [1, 10], [3, 10]>})

// -----

func.func @transpose(%arg0: tensor<?x?x?x?xi32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [1, 2, 3, 4], opt = [5, 6, 7, 9], max = [10, 11, 12, 13]>}) -> tensor<?x?x?x?xi32> {
	%c2 = arith.constant 2 : index
	%c3 = arith.constant 3 : index
	%c0 = arith.constant 0 : index
	%c1 = arith.constant 1 : index
	%0 = stablehlo.transpose %arg0, dims = [1, 0, 3, 2] : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
	%dim = tensor.dim %arg0, %c1 : tensor<?x?x?x?xi32>
	%dim_0 = tensor.dim %arg0, %c0 : tensor<?x?x?x?xi32>
	%dim_1 = tensor.dim %arg0, %c3 : tensor<?x?x?x?xi32>
	%dim_2 = tensor.dim %arg0, %c2 : tensor<?x?x?x?xi32>
	%1 = plan.with_shape %0(%dim, %dim_0, %dim_1, %dim_2) : (tensor<?x?x?x?xi32>, index, index, index, index) -> tensor<?x?x?x?xi32>
	return %1 : tensor<?x?x?x?xi32>
}

// CHECK-LABEL: @transpose
// CHECK-SAME: -> (tensor<?x?x?x?xi32> {tensorrt.shape_profile = #plan.bounds<shape, [2, 1, 4, 3], [11, 10, 13, 12]>})

// -----

func.func @reverse(%arg0: tensor<?x?x?x?xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [1, 2, 3, 4], opt = [5, 6, 7, 9], max = [10, 11, 12, 13]>}) -> tensor<?x?x?x?xf32> {
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = stablehlo.reverse %arg0, dims = [1, 3] : tensor<?x?x?x?xf32>
  %dim = tensor.dim %0, %c0 : tensor<?x?x?x?xf32>
  %dim_0 = tensor.dim %0, %c1 : tensor<?x?x?x?xf32>
  %dim_1 = tensor.dim %0, %c2 : tensor<?x?x?x?xf32>
  %dim_2 = tensor.dim %0, %c3 : tensor<?x?x?x?xf32>
  %1 = plan.with_shape %0(%dim, %dim_0, %dim_1, %dim_2) : (tensor<?x?x?x?xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: @reverse
// CHECK-SAME: -> (tensor<?x?x?x?xf32> {tensorrt.shape_profile = #plan.bounds<shape, [0, 0, 0, 0], [2147483647, 2147483647, 2147483647, 2147483647]>})

// -----

func.func @broadcast(%arg0: tensor<?xi32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [2], opt = [6], max = [10]>}) -> tensor<1x2x?xi32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %0 = stablehlo.broadcast %arg0, sizes = [1, 2] : (tensor<?xi32>) -> tensor<1x2x?xi32>
  %dim = tensor.dim %arg0, %c0 : tensor<?xi32>
  %1 = plan.with_shape %0(%c1, %c2, %dim) : (tensor<1x2x?xi32>, index, index, index) -> tensor<1x2x?xi32>
  return %1 : tensor<1x2x?xi32>
}

// CHECK-LABEL: @broadcast
// CHECK-SAME: -> (tensor<1x2x?xi32> {tensorrt.shape_profile = #plan.bounds<shape, [1, 2, 2], [1, 2, 10]>})

// -----

func.func @gather(%arg0: tensor<3x4x2xi32>, %arg1: tensor<?x3x2xi64> {tensorrt.shape_profile = #tensorrt.shape_profile<min=[1, 3, 2], opt=[2, 3, 2], max=[3, 3, 2]>}) -> tensor<?x3x2x2xi32> {
	%c2 = arith.constant 2 : index
	%c3 = arith.constant 3 : index
	%c0 = arith.constant 0 : index
	%0 = "stablehlo.gather"(%arg0, %arg1) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [1, 0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 2, 2>} : (tensor<3x4x2xi32>, tensor<?x3x2xi64>) -> tensor<?x3x2x2xi32>
	%dim = tensor.dim %arg1, %c0 : tensor<?x3x2xi64>
	%1 = plan.with_shape %0(%dim, %c3, %c2, %c2) : (tensor<?x3x2x2xi32>, index, index, index, index) -> tensor<?x3x2x2xi32>
	return %1 : tensor<?x3x2x2xi32>
}

// CHECK-LABEL: @gather
// CHECK-SAME: -> (tensor<?x3x2x2xi32> {tensorrt.shape_profile = #plan.bounds<shape, [1, 3, 2, 2], [3, 3, 2, 2]>})

// -----

func.func @test_dynamic_reshape(%arg0: tensor<?xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [1], opt = [2], max = [3]>}, %arg1: tensor<2xi32> {tensorrt.value_bounds = #tensorrt.shape_profile<min=[1, 1], opt=[5, 5], max=[40, 40]>}) -> tensor<?x?xf32> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = stablehlo.dynamic_reshape %arg0, %arg1 : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %extracted = tensor.extract %arg1[%c0] : tensor<2xi32>
  %1 = arith.index_cast %extracted : i32 to index
  %extracted_0 = tensor.extract %arg1[%c1] : tensor<2xi32>
  %2 = arith.index_cast %extracted_0 : i32 to index
  %3 = plan.with_shape %0(%1, %2) : (tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}

// CHECK-LABEL: @test_dynamic_reshape
// CHECK-SAME: -> (tensor<?x?xf32> {tensorrt.shape_profile = #plan.bounds<shape, [1, 1], [40, 40]>})

// -----

func.func @test_get_dim_size_max(%arg0: tensor<?x?xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [1, 2], opt = [2, 3], max = [3, 4]>}, %arg1: tensor<?x?xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<min=[1, 1], opt=[30, 50], max=[60, 100]>}) -> tensor<?x?xf32> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<1x1xf32>
  %1 = "stablehlo.get_dimension_size"(%arg0) {dimension = 0 : i64} : (tensor<?x?xf32>) -> tensor<i32>
  %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
  %3 = "stablehlo.get_dimension_size"(%arg0) {dimension = 1 : i64} : (tensor<?x?xf32>) -> tensor<i32>
  %4 = stablehlo.reshape %3 : (tensor<i32>) -> tensor<1xi32>
  %5 = "stablehlo.concatenate"(%2, %4) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %6 = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x?xf32>) -> tensor<i32>
  %7 = stablehlo.reshape %6 : (tensor<i32>) -> tensor<1xi32>
  %8 = "stablehlo.get_dimension_size"(%arg1) {dimension = 1 : i64} : (tensor<?x?xf32>) -> tensor<i32>
  %9 = stablehlo.reshape %8 : (tensor<i32>) -> tensor<1xi32>
  %10 = "stablehlo.concatenate"(%7, %9) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %11 = stablehlo.maximum %5, %10 : tensor<2xi32>
  %12 = stablehlo.dynamic_broadcast_in_dim %0, %11, dims=[0, 1] : (tensor<1x1xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg1, %c0 : tensor<?x?xf32>
  %13 = arith.maxsi %dim, %dim_0 : index
  %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %dim_2 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %14 = arith.maxsi %dim_1, %dim_2 : index
  %15 = plan.with_shape %12(%13, %14) : (tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
  return %15 : tensor<?x?xf32>
}

// CHECK-LABEL: @test_get_dim_size_max
// CHECK-SAME: -> (tensor<?x?xf32> {tensorrt.shape_profile = #plan.bounds<shape, [1, 2], [60, 100]>})

// -----

func.func @dot_general(%arg0: tensor<?x?x?xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [1, 2, 2], opt = [1, 4, 2], max = [1, 4, 4]>}, %arg1: tensor<?x?x?xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [1, 2, 3], opt = [2, 3, 4], max = [3, 4, 5]>}) -> tensor<?x?x?xf32> {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dim = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %dim_0 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %dim_1 = tensor.dim %arg1, %c2 : tensor<?x?x?xf32>
  %1 = plan.with_shape %0(%dim, %dim_0, %dim_1) : (tensor<?x?x?xf32>, index, index, index) -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}

// CHECK-LABEL: @dot_general
// CHECK-SAME: -> (tensor<?x?x?xf32> {tensorrt.shape_profile = #plan.bounds<shape, [1, 2, 3], [1, 4, 5]>})

// -----

func.func @test_loop_concat(%arg0: tensor<1xf32>, %arg1: tensor<1xi32> {tensorrt.value_bounds = #tensorrt.shape_profile<min = [1], opt = [2], max = [4]>}, %arg2: tensor<?xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [2], opt = [4], max = [6]>}, %arg3: tensor<1024xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg2, %c0 : tensor<?xf32>
  %extracted = tensor.extract %arg1[%c0] : tensor<1xi32>
  %0 = arith.index_cast %extracted : i32 to index
  %1 = scf.for %arg4 = %dim to %0 step %c1 iter_args(%arg5 = %arg2) -> (tensor<?xf32>) {
    %3 = "stablehlo.concatenate"(%arg5, %arg0) {dimension = 0 : i64} : (tensor<?xf32>, tensor<1xf32>) -> tensor<?xf32>
    %dim_1 = tensor.dim %arg5, %c0 : tensor<?xf32>
    %4 = arith.addi %dim_1, %c1 : index
    %5 = plan.with_shape %3(%4) : (tensor<?xf32>, index) -> tensor<?xf32>
    scf.yield %5 : tensor<?xf32>
  }
  %dim_0 = tensor.dim %1, %c0 : tensor<?xf32>
  %2 = plan.with_shape %1(%dim_0) : (tensor<?xf32>, index) -> tensor<?xf32>
  return %2 : tensor<?xf32>
}

// CHECK-LABEL: @test_loop_concat
// CHECK-SAME: -> (tensor<?xf32> {tensorrt.shape_profile = #plan.bounds<shape, [0], [2147483647]>})

// -----

func.func @real_dynamic_slice(%arg0: tensor<?xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [1], opt = [2], max = [4]>}, %arg1: tensor<1xindex> { tensorrt.value_bounds = #tensorrt.shape_profile<min = [0], opt = [0], max = [0]>}, %arg2: tensor<1xindex> { tensorrt.value_bounds = #tensorrt.shape_profile<min = [3], opt = [4], max = [5]>}, %arg3: tensor<1xindex> { tensorrt.value_bounds = #tensorrt.shape_profile<min = [1], opt = [1], max = [1]>}) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = stablehlo.real_dynamic_slice %arg0, %arg1, %arg2, %arg3 : (tensor<?xf32>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>) -> tensor<?xf32>
  %extracted = tensor.extract %arg1[%c0] : tensor<1xindex>
  %extracted_0 = tensor.extract %arg2[%c0] : tensor<1xindex>
  %extracted_1 = tensor.extract %arg3[%c0] : tensor<1xindex>
  %1 = arith.subi %extracted_0, %extracted : index
  %2 = arith.addi %extracted_1, %1 : index
  %3 = arith.subi %2, %c1 : index
  %4 = arith.divsi %3, %extracted_1 : index
  %5 = plan.with_shape %0(%4) : (tensor<?xf32>, index) -> tensor<?xf32>
  return %5 : tensor<?xf32>
}

// CHECK-LABEL: @real_dynamic_slice
// CHECK-SAME: -> (tensor<?xf32> {tensorrt.shape_profile = #plan.bounds<shape, [3], [5]>})

// -----

#bounds0 = #tensorrt.shape_profile<min=[10], opt=[20], max=[30]>
#bounds1 = #tensorrt.shape_profile<min=[2,2], opt=[5,5], max=[10,10]>

func.func @value_bounds(%arg0: tensor<?xf32> {tensorrt.shape_profile = #bounds0}, %arg1: tensor<2xi32> {tensorrt.value_bounds = #bounds1}) -> (tensor<?x?xf32>, tensor<2xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.extract %arg1[%c0] : tensor<2xi32>
  %d1 = tensor.extract %arg1[%c1] : tensor<2xi32>
  %with_bounds = plan.with_values {tag="with_values"} %arg1(%d0, %d1) : tensor<2xi32>
  %0 = stablehlo.dynamic_reshape %arg0, %with_bounds : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %1 = plan.with_shape %0 (%d0, %d1) : (tensor<?x?xf32>, i32, i32) -> (tensor<?x?xf32>)
  return {tag="return"} %0, %with_bounds : tensor<?x?xf32>, tensor<2xi32>
}

// CHECK-LABEL: func.func @value_bounds
//  CHECK-SAME:   tensor<?xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [10], opt = [20], max = [30]>}
//  CHECK-SAME:   tensor<2xi32> {tensorrt.value_bounds = #tensorrt.shape_profile<min = [2, 2], opt = [5, 5], max = [10, 10]>})
//  CHECK-SAME:  -> (tensor<?x?xf32> {tensorrt.shape_profile = #plan.bounds<shape, [2, 2], [10, 10]>},
//  CHECK-SAME:   tensor<2xi32> {tensorrt.value_bounds = #plan.bounds<value, dense<2> : tensor<2xi64>, dense<10> : tensor<2xi64>>}
