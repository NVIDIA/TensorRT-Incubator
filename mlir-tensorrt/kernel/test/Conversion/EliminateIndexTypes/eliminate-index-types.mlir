// RUN: kernel-opt %s -kernel-eliminate-index-types="index-bitwidth=32" --split-input-file | FileCheck %s

func.func @index_ops(%a: index, %b: tensor<2xindex>) -> i32 {
  %c4 = arith.constant 4 : index
  %dense = arith.constant dense<1> : tensor<2xindex>
  %sum = arith.addi %a, %c4 : index
  %cast = arith.index_cast %sum : index to i32
  return %cast : i32
}

// CHECK-LABEL: func.func @index_ops
// CHECK-SAME: (%[[A:.+]]: i32, %[[B:.+]]: tensor<2xi32>) -> i32
// CHECK-DAG:    %[[C4:.+]] = arith.constant 4 : i32
// CHECK-DAG:    %[[DENSE:.+]] = arith.constant dense<1> : tensor<2xi32>
// CHECK-DAG:    %[[SUM:.+]] = arith.addi %[[A]], %[[C4]] : i32
// CHECK-DAG:    return %[[SUM]] : i32

// -----

func.func @loop(%lb: index, %ub: index, %step: index) {
  scf.for %i = %lb to %ub step %step {
    scf.yield
  }
  return
}

// CHECK-LABEL: func.func @loop
// CHECK-SAME: (%[[LB:.+]]: i32, %[[UB:.+]]: i32, %[[STEP:.+]]: i32)
// CHECK: scf.for %[[IV:.+]] = %[[LB]] to %[[UB]] step %[[STEP]]
// CHECK-NOT: index

// -----

func.func @loop_iv_used(%lb: index, %ub: index, %step: index) {
  scf.for %i = %lb to %ub step %step {
    %one = arith.constant 1 : index
    %add = arith.addi %i, %one : index
    scf.yield
  }
  return
}

// CHECK-LABEL: func.func @loop_iv_used
// CHECK-SAME: (%[[LB:.+]]: i32, %[[UB:.+]]: i32, %[[STEP:.+]]: i32)
// CHECK: scf.for %[[IV:.+]] = %[[LB]] to %[[UB]] step %[[STEP]]
// CHECK: %[[ONE:.+]] = arith.constant 1 : i32
// CHECK: %[[ADD:.+]] = arith.addi %[[IV]], %[[ONE]] : i32
// CHECK-NOT: index
// CHECK-NOT: index
