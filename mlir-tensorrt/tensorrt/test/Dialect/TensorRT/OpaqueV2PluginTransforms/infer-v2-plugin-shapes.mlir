// RUN: tensorrt-opt %s -split-input-file -tensorrt-infer-plugin-shapes | FileCheck %s

// This test will load the plugin creator from the registry, instantiate the plugin, then
// we provide a special implementation of the IExprsBuilder to the plugin's
// `getOutputDimensions` function. The special implementation constructs the scalar shape
// calculation in the plugin's shape region.

func.func @test_shape_region_v2_populate(%arg0: tensor<?x4x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = tensorrt.opaque_plugin {
    dso_path = "TensorRTTestPlugins.so",
    plugin_name = "TestV2InferShapePlugin",
    plugin_version = "0",
    plugin_namespace = "",
    creator_params = {}
  } (%arg0) : (tensor<?x4x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// Note that the specifics of what this region should contain should be derived from
// `test/lib/Target/TensorRT/TestPlugins.cpp`.

// CHECK-LABEL: @test_shape_region_v2_populate
//       CHECK:     %[[v0:.+]] = tensorrt.opaque_plugin
//  CHECK-NEXT:     ^bb0(%[[arg1:.+]]: i64, %[[arg2:.+]]: i64, %[[arg3:.+]]: i64, %[[arg4:.+]]: i64):
//  CHECK-NEXT:       %[[c4_i64:.+]] = arith.constant 4 : i64
//  CHECK-NEXT:       %[[c42_i64:.+]] = arith.constant 42 : i64
//  CHECK-NEXT:       %[[v1:.+]] = arith.addi %[[arg1]], %[[c4_i64]] : i64
//  CHECK-NEXT:       %[[v2:.+]] = arith.maxsi %[[c4_i64]], %[[arg3]] : i64
//  CHECK-NEXT:       %[[c3_i64:.+]] = arith.constant 3 : i64
//  CHECK-NEXT:       %[[v3:.+]] = arith.ceildivsi %[[arg4]], %[[c3_i64]] : i64
//  CHECK-NEXT:       tensorrt.yield %[[c42_i64]], %[[v1]], %[[v2]], %[[v3]] : i64, i64, i64, i64
