// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

// CHECK-LABEL: @dynamic_nd_iota_3
//  CHECK-SAME: tensorrt.engine
func.func @dynamic_nd_iota_3(%arg0: tensor<2xi32> {
  tensorrt.value_bounds = #tensorrt.shape_profile<min=[1, 3], opt=[4, 3], max=[12, 3]>,
  tensorrt.host_tensor
}) -> tensor<?x3xi64> {
  %cst_f16 = tensorrt.constant dense<0> : tensor<i64>
  %cst_f16_0 = tensorrt.constant dense<[0, 1]> : tensor<2xi64>
  %0 = tensorrt.linspace[%cst_f16 : tensor<i64>] [%arg0 : tensor<2xi32>] [%cst_f16_0 : tensor<2xi64>] : tensor<?x3xi64>
  return %0 : tensor<?x3xi64>
}
