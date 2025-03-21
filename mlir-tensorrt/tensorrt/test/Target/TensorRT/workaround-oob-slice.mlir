// RUN: %pick-one-gpu tensorrt-opt -pass-pipeline="builtin.module(tensorrt-apply-bug-wars{force-default-slice-in-bounds},translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

// This shape profile reflects that we don't have good information about %arg1 bounds.
// It will cause a build failure if the slice offset is not constrained using `offset = max(0, arg1)`.
#bounds = #tensorrt.shape_profile<min=[-2147483648, -2147483648], opt=[-1, -1], max=[2147483647, 2147483647]>

func.func @tensorrt_default_slice_to_clamp(%arg0: tensor<10x10xf32>,
                                           %arg1: tensor<2xi32> {
                                                      tensorrt.host_tensor,
                                                      tensorrt.value_bounds = #bounds}
                                          ) -> tensor<10x10xf32> {
  %0 = tensorrt.slice %arg0[%arg1 : tensor<2xi32>][10, 10][1, 1] : tensor<10x10xf32> to tensor<10x10xf32>
  return %0 : tensor<10x10xf32>
}

// CHECK-LABEL: @tensorrt_default_slice_to_clamp
//  CHECK-SAME: tensorrt.engine