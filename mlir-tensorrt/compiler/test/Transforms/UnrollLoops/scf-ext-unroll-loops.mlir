// RUN: mlir-tensorrt-opt %s -mtrt-scf-unroll=unroll-threshold=99 -split-input-file \
// RUN: | FileCheck %s --check-prefix=T99

// RUN: mlir-tensorrt-opt %s -mtrt-scf-unroll=unroll-threshold=100 -split-input-file \
// RUN: | FileCheck %s --check-prefix=T100

// The unrolling transformation is tested upstream. This test just checks that the options work
// correctly.

// T99-LABEL: func @unroll_for_loop_with_static_trip_count
// T100-LABEL: func @unroll_for_loop_with_static_trip_count
func.func @unroll_for_loop_with_static_trip_count(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>) -> tensor<100xf32> {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c1 = arith.constant 1 : index
  //      T99: scf.for
  // T100-NOT: scf.for
  // T100-COUNT-100: arith.addf
  %0 = scf.for %arg2 = %c0 to %c100 step %c1 iter_args(%arg3 = %arg0) -> (tensor<100xf32>) {
    %1 = arith.addf %arg3, %arg1 : tensor<100xf32>
    scf.yield %1 : tensor<100xf32>
  }
  return %0 : tensor<100xf32>
}
