// RUN: tensorrt-opt %s -split-input-file -tensorrt-infer-plugin-shapes | FileCheck %s
// RUN: tensorrt-opt %s -split-input-file -tensorrt-infer-plugin-shapes -test-tensorrt-shape-inference | FileCheck %s --check-prefix=REIFY

// This test will load the plugin creator from the registry, instantiate the plugin, then
// we provide a special implementation of the IExprsBuilder to the plugin's
// `getOutputShapes` function. The special implementation constructs the scalar shape
// calculation in the plugin's shape region.

func.func @test_quick_plugin_shape_region_populate(%arg0: tensor<?x4x?x?xf32>) -> (tensor<?x?x?x?xf32>, index, index, index, index) {
  %0 = tensorrt.opaque_plugin {
    dso_path = "TensorRTTestPlugins.so",
    plugin_name = "TestQuickPluginShape",
    plugin_version = "0",
    plugin_namespace = "",
    creator_params = {}
  } (%arg0) : (tensor<?x4x?x?xf32>) -> tensor<?x?x?x?xf32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %d0 = tensor.dim %0, %c0 : tensor<?x?x?x?xf32>
  %d1 = tensor.dim %0, %c1 : tensor<?x?x?x?xf32>
  %d2 = tensor.dim %0, %c2 : tensor<?x?x?x?xf32>
  %d3 = tensor.dim %0, %c3 : tensor<?x?x?x?xf32>

  return %0, %d0, %d1, %d2, %d3 : tensor<?x?x?x?xf32>, index, index, index, index
}

// CHECK-LABEL: Input ranks: 4
// Dynamic dimensions should use INT64_MIN
// CHECK-LABEL: Input DimExprs Constant Values:
// CHECK-NEXT: Dimension 0 value: -9223372036854775808
// CHECK-NEXT: Dimension 1 value: 4
// CHECK-NEXT: Dimension 2 value: -9223372036854775808
// CHECK-NEXT: Dimension 3 value: -9223372036854775808

// Note that the specifics of what this region should contain should be derived from
// `test/lib/Target/TensorRT/TestPlugins.cpp`.

// CHECK-LABEL: @test_quick_plugin_shape_region_populate
//       CHECK:     %[[v0:.+]] = tensorrt.opaque_plugin
//  CHECK-NEXT:     ^bb0(%[[arg1:.+]]: i64, %[[arg2:.+]]: i64, %[[arg3:.+]]: i64, %[[arg4:.+]]: i64):
//  CHECK-NEXT:       %[[c4_i64:.+]] = arith.constant 4 : i64
//  CHECK-NEXT:       %[[c42_i64:.+]] = arith.constant 42 : i64
//  CHECK-NEXT:       %[[v1:.+]] = arith.addi %[[arg1]], %[[c4_i64]] : i64
//  CHECK-NEXT:       %[[v2:.+]] = arith.maxsi %[[c4_i64]], %[[arg3]] : i64
//  CHECK-NEXT:       %[[c3_i64:.+]] = arith.constant 3 : i64
//  CHECK-NEXT:       %[[v3:.+]] = arith.ceildivsi %[[arg4]], %[[c3_i64]] : i64
//  CHECK-NEXT:       tensorrt.yield %[[c42_i64]], %[[v1]], %[[v2]], %[[v3]] : i64, i64, i64, i64

// REIFY-LABEL: @test_quick_plugin_shape_region_populate
//  REIFY-SAME: (%[[arg0:.+]]: tensor<?x4x?x?xf32>)
//   REIFY-DAG:     %[[c3:.+]] = arith.constant 3 : index
//   REIFY-DAG:     %[[c3_i64:.+]] = arith.constant 3 : i64
//   REIFY-DAG:     %[[c2:.+]] = arith.constant 2 : index
//   REIFY-DAG:     %[[c4_i64:.+]] = arith.constant 4 : i64
//   REIFY-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   REIFY-DAG:     %[[c42:.+]] = arith.constant 42 : index
//       REIFY:     %[[v0:.+]] = tensorrt.opaque_plugin {creator_params = {}, dso_path = "TensorRTTestPlugins.so", plugin_name = "TestQuickPluginShape", plugin_namespace = "", plugin_version = "0"}(%[[arg0]]) : (tensor<?x4x?x?xf32>) -> tensor<?x?x?x?xf32> {
//                     --- no need to check all the ops in the body again ---
//       REIFY:       tensorrt.yield
//  REIFY-NEXT:     }
//   REIFY-DAG:     %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]] : tensor<?x4x?x?xf32>
//   REIFY-DAG:     %[[v1:.+]] = arith.index_cast %[[dim]] : index to i64
//   REIFY-DAG:     %[[v2:.+]] = arith.addi %[[v1]], %[[c4_i64]] : i64
//   REIFY-DAG:     %[[v3:.+]] = arith.index_cast %[[v2]] : i64 to index
//   REIFY-DAG:     %[[dim_0:.+]] = tensor.dim %[[arg0]], %[[c2]] : tensor<?x4x?x?xf32>
//   REIFY-DAG:     %[[v4:.+]] = arith.index_cast %[[dim_0]] : index to i64
//   REIFY-DAG:     %[[v5:.+]] = arith.maxsi %[[v4]], %[[c4_i64]] : i64
//   REIFY-DAG:     %[[v6:.+]] = arith.index_cast %[[v5]] : i64 to index
//   REIFY-DAG:     %[[dim_1:.+]] = tensor.dim %[[arg0]], %[[c3]] : tensor<?x4x?x?xf32>
//   REIFY-DAG:     %[[v7:.+]] = arith.index_cast %[[dim_1]] : index to i64
//   REIFY-DAG:     %[[v8:.+]] = arith.ceildivsi %[[v7]], %[[c3_i64]] : i64
//   REIFY-DAG:     %[[v9:.+]] = arith.index_cast %[[v8]] : i64 to index
//   REIFY-DAG:     return %[[v0]], %[[c42]], %[[v3]], %[[v6]], %[[v9]] :
