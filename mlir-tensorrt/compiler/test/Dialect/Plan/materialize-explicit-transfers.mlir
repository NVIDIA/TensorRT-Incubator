// RUN: mlir-tensorrt-opt %s --plan-materialize-explicit-transfers -split-input-file | FileCheck %s

!host_type = tensor<1xf32, #plan.memory_space<host>>
!device_type = tensor<1xf32, #plan.memory_space<device>>

// CHECK-LABEL: func.func @tensor_cast
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xf32, #plan.memory_space<host>>
func.func @tensor_cast(%arg0: !host_type) -> !device_type {
  //      CHECK: %[[v0:.+]] = tensor.empty() {{.*}} #plan.memory_space<device>
  //      CHECK: %[[v1:.+]] = bufferization.materialize_in_destination
  // CHECK-SAME:   %[[arg0]] in %[[v0]]
  %1 = tensor.cast %arg0 : !host_type to !device_type
  //      CHECK: return %[[v1]] : tensor<1xf32, #plan.memory_space<device>>
  return %1 : !device_type
}

// -----

!host_type = tensor<?x4x?xf32, #plan.memory_space<host>>
!device_type = tensor<?x4x?xf32, #plan.memory_space<device>>

// CHECK-LABEL: func.func @dynamic_shape
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x4x?xf32, #plan.memory_space<host>>
func.func @dynamic_shape(%arg0: !host_type) -> !device_type {
//    CHECK-DAG: %[[c2:.+]] = arith.constant 2 : index
//    CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
//    CHECK-DAG: %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]] :
//    CHECK-DAG: %[[dim_0:.+]] = tensor.dim %[[arg0]], %[[c2]] :
  //      CHECK: %[[v0:.+]] = tensor.empty(%[[dim]], %[[dim_0]])
  // CHECK-SAME:  tensor<?x4x?xf32, #plan.memory_space<device>>

  //      CHECK: %[[v1:.+]] = bufferization.materialize_in_destination
  // CHECK-SAME:   %[[arg0]] in %[[v0]]
  %1 = tensor.cast %arg0 : !host_type to !device_type
  //      CHECK: return %[[v1]] : tensor<?x4x?xf32, #plan.memory_space<device>>
  return %1 : !device_type
}


// -----

!host_type = tensor<4xf32, #plan.memory_space<host>>
!device_type = tensor<4xf32, #plan.memory_space<device>>

func.func @redundant_materialize_in_dest(%arg0: !device_type) -> !host_type {
  %0 = tensor.empty() : !host_type
  %1 = bufferization.materialize_in_destination %arg0 in %0
    : (!device_type, !host_type) -> !host_type
  %2 = bufferization.materialize_in_destination %arg0 in %1
    : (!device_type, !host_type) -> !host_type
  return %2 : !host_type
}

// CHECK-LABEL: func.func @redundant_materialize_in_dest
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xf32, #plan.memory_space<device>>) -> tensor<4xf32, #plan.memory_space<host>> {
//       CHECK:     %[[v0:.+]] = tensor.empty() : tensor<4xf32, #plan.memory_space<host>>
//       CHECK:     %[[v1:.+]] = bufferization.materialize_in_destination %[[arg0]] in %[[v0]]
//       CHECK:     return %[[v1]] : tensor<4xf32, #plan.memory_space<host>>
