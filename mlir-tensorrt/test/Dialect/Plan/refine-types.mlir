// RUN: mlir-tensorrt-opt %s -split-input-file -plan-refine-types | FileCheck %s

func.func @refine_dynamic_broadcast_requires_cast(%arg0: tensor<?xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %c1024 = arith.constant 1024 : i32
  %c512 = arith.constant 512 : i32
  %0 = plan.with_values %arg1 (%c1024, %c512) : tensor<2xi32>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims = [1] : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @refine_dynamic_broadcast_requires_cast({{.*}}) -> tensor<1024x512xf32> {
//       CHECK:     %[[v1:.+]] = stablehlo.dynamic_broadcast_in_dim {{.*}} -> tensor<1024x512xf32>
//       CHECK:     return %[[v1]] : tensor<1024x512xf32>

// -----

func.func @fold_identity_broadcast(%arg0: tensor<?x?xf32>, %arg1: tensor<2xi32>, %arg2: i32, %arg3: i32) -> tensor<?x?xf32> {
  %arg0_with_hint = plan.with_shape %arg0 (%arg2, %arg3) : (tensor<?x?xf32>, i32, i32) -> tensor<?x?xf32>
  %arg1_with_hint = plan.with_values %arg1 (%arg2, %arg3) : tensor<2xi32>
  %0 = stablehlo.dynamic_broadcast_in_dim %arg0_with_hint, %arg1_with_hint, dims = [0, 1] : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @fold_identity_broadcast
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?xf32>, {{.*}}) -> tensor<?x?xf32>
//  CHECK-NEXT:     return %[[arg0]] : tensor<?x?xf32>

// -----

func.func @fold_identity_broadcast_requires_cast(%arg0: tensor<1x?xf32>, %arg1: tensor<2xi32>, %arg2: i32, %arg3: i32) -> tensor<?x?xf32> {
  %arg0_with_hint = plan.with_shape %arg0 (%arg2, %arg3) : (tensor<1x?xf32>, i32, i32) -> tensor<1x?xf32>
  %arg1_with_hint = plan.with_values %arg1 (%arg2, %arg3) : tensor<2xi32>
  %0 = stablehlo.dynamic_broadcast_in_dim %arg0_with_hint, %arg1_with_hint, dims = [0, 1] : (tensor<1x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @fold_identity_broadcast_requires_cast
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x?xf32>, {{.*}}) -> tensor<1x?xf32>
//  CHECK-NEXT:     return %[[arg0]] : tensor<1x?xf32>

// -----

func.func @refine_dynamic_broadcast_add(%arg0: tensor<?xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %c1024 = arith.constant 1024 : i32
  %c512 = arith.constant 512 : i32
  %0 = plan.with_values %arg1 (%c1024, %c512) : tensor<2xi32>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims = [1] : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %2 = stablehlo.add %1, %1 : tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @refine_dynamic_broadcast_add
//  CHECK-SAME:  -> tensor<1024x512xf32> {
//       CHECK:     %[[v1:.+]] = stablehlo.dynamic_broadcast_in_dim {{.*}} -> tensor<1024x512xf32>
//       CHECK:     %[[v2:.+]] = stablehlo.add %[[v1]], %[[v1]] : tensor<1024x512xf32>
//       CHECK:     return %[[v2]] : tensor<1024x512xf32>

// -----

func.func @refine_dynamic_iota_requires_cast(%arg0: tensor<1xi32>) -> tensor<?xi32> {
  %c1024 = arith.constant 1024 : i32
  %0 = plan.with_values %arg0 (%c1024) : tensor<1xi32>
  %1 = "stablehlo.dynamic_iota"(%0) {iota_dimension = 0 : i64} : (tensor<1xi32>) -> tensor<?xi32>
  return %1 : tensor<?xi32>
}

// CHECK-LABEL: func.func @refine_dynamic_iota_requires_cast
//  CHECK-SAME:  -> tensor<1024xi32> {
//       CHECK:     %[[v1:.+]] = stablehlo.dynamic_iota {{.*}} -> tensor<1024xi32>
//       CHECK:     return %[[v1]] : tensor<1024xi32>


// -----

func.func @refine_add_with_shape(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %c1024 = arith.constant 1024 : i32
  %1 = stablehlo.add %arg0, %arg1 : tensor<?xf32>
  %2 = plan.with_shape %1 (%c1024) : (tensor<?xf32>, i32) ->  tensor<?xf32>
  return %2 : tensor<?xf32>
}

// CHECK-LABEL: func.func @refine_add_with_shape
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xf32>, %[[arg1:.+]]: tensor<?xf32>)
//       CHECK:     %[[v0:.+]] = stablehlo.add %[[arg0]], %[[arg1]] :
//       CHECK:     return %[[v0]] : tensor<1024xf32>


// -----

func.func @reverse(%arg0: tensor<?x?x1024x?xf32>) -> tensor<?x?x?x?xf32> {
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = stablehlo.reverse %arg0, dims = [1, 3] : tensor<?x?x1024x?xf32>
  %dim = tensor.dim %arg0, %c0 : tensor<?x?x1024x?xf32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?x1024x?xf32>
  %dim_1 = tensor.dim %arg0, %c2 : tensor<?x?x1024x?xf32>
  %dim_2 = tensor.dim %arg0, %c3 : tensor<?x?x1024x?xf32>
  %00 = tensor.cast %0 : tensor<?x?x1024x?xf32> to tensor<?x?x?x?xf32>
  %1 = plan.with_shape %00(%dim, %dim_0, %dim_1, %dim_2) : (tensor<?x?x?x?xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: func.func @reverse
//  CHECK-SAME: ({{.*}}) -> tensor<?x?x1024x?xf32>
//       CHECK-DAG:     %[[v1:.+]] = plan.with_shape
//       CHECK-DAG:     return %[[v1]] : tensor<?x?x1024x?xf32>


// -----

func.func @tensorrt_opaque_plugin_no_cast() -> tensor<?xf32> {
  %c1 = arith.constant 1 : index
  %c64_i32 = arith.constant 64 : i32
  %cst = stablehlo.constant dense<1.000000e+00> : tensor<64xf32>
  %c = stablehlo.constant dense<64> : tensor<1xi32>
  %0 = stablehlo.dynamic_reshape %cst, %c : (tensor<64xf32>, tensor<1xi32>) -> tensor<?xf32>
  %1 = plan.with_shape %0(%c64_i32) : (tensor<?xf32>, i32) -> tensor<?xf32>
  %2 = tensorrt.opaque_plugin {creator_func = "getTestV2Plugin1Creator", creator_params = {i16_param = 20 : i16, i32_param = 10 : i32, i64_param = 31 : i64, i8_param = 40 : i8, shape_param = array<i64: 1, 2, 3>}, dso_path = "libTensorRTTestPlugins.so", plugin_name = "TestV2Plugin1", plugin_namespace = "", plugin_version = "0"}(%1) : (tensor<?xf32>) -> tensor<?xf32> {
  ^bb0(%arg0: i64):
    %c1_i64 = arith.constant 1 : i64
    tensorrt.yield %c1_i64 : i64
  }
  %3 = plan.with_shape %2(%c1) : (tensor<?xf32>, index) -> tensor<?xf32>
  return %3 : tensor<?xf32>
}
// CHECK-LABEL: func.func @tensorrt_opaque_plugin_no_cast
// CHECK-SAME: () -> tensor<?xf32>
// CHECK: %[[v1:.*]] = stablehlo.dynamic_reshape %{{.*}}, %{{.*}} : (tensor<64xf32>, tensor<1xi32>) -> tensor<64xf32>
// CHECK: %[[v2:.*]] = tensorrt.opaque_plugin
// CHECK-SAME: (%[[v1]]) : (tensor<64xf32>) -> tensor<?xf32>
// CHECK: return %{{.*}} : tensor<?xf32>