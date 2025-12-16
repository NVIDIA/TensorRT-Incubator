// RUN: kernel-opt %s -split-input-file -transform-interpreter -canonicalize | FileCheck %s

#map15 = affine_map<() -> ()>
func.func @test_scalar_linalg(%arg0: tensor<i32>, %arg1: tensor<i8>, %arg2: tensor<i8>) -> tensor<i8> {
  %c1_i8 = arith.constant 1 : i8
  %c20_i32 = arith.constant 20 : i32
  %c-1_i8 = arith.constant -1 : i8
  %0 = linalg.generic {indexing_maps = [#map15, #map15, #map15], iterator_types = []} ins(%arg0, %arg1 : tensor<i32>, tensor<i8>) outs(%arg2 : tensor<i8>) {
  ^bb0(%in: i32, %in_0: i8, %out: i8):
    %1 = arith.cmpi eq, %in, %c20_i32 : i32
    %2 = arith.extui %1 : i1 to i8
    %3 = arith.ori %2, %in_0 : i8
    %4 = arith.xori %3, %c-1_i8 : i8
    %5 = arith.andi %4, %c1_i8 : i8
    linalg.yield %5 : i8
  } -> tensor<i8>
  return %0 : tensor<i8>
}

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0: (!transform.any_op) -> !transform.any_op
    %forall_op, %linalg_op = transform.kernel.nest_scalar_linalg_in_forall %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.kernel.forall_to_kernel %forall_op threads(1) {
      gpu_target = #nvvm.target<chip = "sm_80">
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: func.func @test_scalar_linalg
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i32>, %[[arg1:.+]]: tensor<i8>, %[[arg2:.+]]: tensor<i8>) -> tensor<i8> {
//       CHECK:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[v0:.+]] = kernel.call @kernels::@test_scalar_linalg_kernel grid[%[[c1]], %[[c1]], %[[c1]]] block[%[[c1]]] (%[[arg0]], %[[arg1]]) outs(%[[arg2]])
//       CHECK:     return %[[v0]] : tensor<i8>
//       CHECK:   gpu.module @kernels
//       CHECK:   func.func @test_scalar_linalg_kernel
//       CHECK:       linalg.generic
