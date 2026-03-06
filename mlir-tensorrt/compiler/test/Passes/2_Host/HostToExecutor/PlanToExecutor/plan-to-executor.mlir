// RUN: mlir-tensorrt-opt %s -split-input-file -convert-plan-to-executor | FileCheck %s

func.func @convert_mem_space(%arg0: tensor<4xindex, #plan.memory_space<host>>)
    -> tensor<4xindex, #plan.memory_space<host>> {
  return %arg0: tensor<4xindex, #plan.memory_space<host>>
}

// CHECK-LABEL: @convert_mem_space
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xindex, #executor.memory_type<host>>) -> tensor<4xindex, #executor.memory_type<host>>
//       CHECK:     return %[[arg0]] : tensor<4xindex, #executor.memory_type<host>>

// -----

func.func @convert_unk_mem_space(%arg0: tensor<4xindex, #plan.memory_space<unknown>>)
    -> tensor<4xindex, #plan.memory_space<unknown>> {
  return %arg0: tensor<4xindex, #plan.memory_space<unknown>>
}

// CHECK-LABEL: @convert_unk_mem_space
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xindex>) -> tensor<4xindex>
//       CHECK:     return %[[arg0]] : tensor<4xindex>

// -----

func.func @convert_from_elements(%arg0: index)
    -> tensor<1xindex, #plan.memory_space<host>> {
  %0 = tensor.from_elements %arg0 : tensor<1xindex, #plan.memory_space<host>>
  return %0 : tensor<1xindex, #plan.memory_space<host>>
}

// CHECK-LABEL: @convert_from_elements
//  CHECK-SAME: (%[[arg0:.+]]: index) -> tensor<1xindex, #executor.memory_type<host>>
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[arg0]] : tensor<1xindex, #executor.memory_type<host>>
//       CHECK:     return %[[from_elements]] : tensor<1xindex, #executor.memory_type<host>>

// -----

func.func @convert_loop(%arg0: index)
    -> tensor<1xindex, #plan.memory_space<host>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant dense<1> : tensor<1xindex, #plan.memory_space<host>>
  %empty = tensor.from_elements %arg0 : tensor<1xindex, #plan.memory_space<host>>
  %0 = scf.for %i = %c0 to %c10 step %c1 iter_args(%iter = %empty)
    -> tensor<1xindex, #plan.memory_space<host>> {
    %1 = arith.addi %iter, %cst : tensor<1xindex, #plan.memory_space<host>>
    scf.yield %1 : tensor<1xindex, #plan.memory_space<host>>
  }
  return %0 : tensor<1xindex, #plan.memory_space<host>>
}

// CHECK-LABEL: @convert_loop
//  CHECK-SAME: (%[[arg0:.+]]: index) -> tensor<1xindex, #executor.memory_type<host>>
//   CHECK-NOT: #plan.memory_space

// -----

func.func @convert_constant() -> tensor<2xf32, #plan.memory_space<host>> {
  %0 = arith.constant dense<[1.0, 2.0]> : tensor<2xf32, #plan.memory_space<host>>
  return %0 : tensor<2xf32, #plan.memory_space<host>>
}

// CHECK-LABEL: @convert_constant
//  CHECK-SAME: () -> tensor<2xf32, #executor.memory_type<host>> {
//       CHECK:     %[[cst:.+]] = arith.constant dense<[{{.+}}]> : tensor<2xf32, #executor.memory_type<host>>
//       CHECK:     return %[[cst]] : tensor<2xf32, #executor.memory_type<host>>

// -----

func.func @convert_extract(%arg0: tensor<2xf32, #plan.memory_space<host>>) -> f32 {
  %c0 = arith.constant 0 : index
  %0 = tensor.extract %arg0[%c0] : tensor<2xf32, #plan.memory_space<host>>
  return %0 : f32
}

// CHECK-LABEL: func.func @convert_extract
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2xf32, #executor.memory_type<host>>)
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c0]]]
//       CHECK:     return %[[extracted]] : f32

// -----

func.func @bounds_attr_conversion(%arg0: tensor<?x?xf32> {plan.shape_bounds = #plan.bounds<shape, [10, 10], [30, 30]>},
                        %arg1: tensor<2xi32> {plan.value_bounds = #plan.bounds<value, dense<[2, 2]>:tensor<2xi32>,dense<[10, 10]>:tensor<2xi32>>})
                        -> (tensor<?x?xf32> {plan.shape_bounds = #plan.bounds<shape, [10, 10], [30, 30]>},
                            tensor<2xi32> {plan.value_bounds = #plan.bounds<value, dense<2> : tensor<1xi32>, dense<10> : tensor<1xi32>>}) {
  return %arg0, %arg1 : tensor<?x?xf32>, tensor<2xi32>
}

// CHECK-LABEL: func.func @bounds_attr_conversion
//  CHECK-SAME: tensor<?x?xf32> {executor.shape_profile = #executor.dim_bounds<min = [10, 10], max = [30, 30]>}
//  CHECK-SAME: tensor<2xi32> {executor.value_bounds = #executor.value_bounds<min = dense<2> : tensor<2xi32>, max = dense<10> : tensor<2xi32>>})
//  CHECK-SAME:  (tensor<?x?xf32> {executor.shape_profile = #executor.dim_bounds<min = [10, 10], max = [30, 30]>}
//  CHECK-SAME:   tensor<2xi32> {executor.value_bounds = #executor.value_bounds<min = dense<2> : tensor<1xi32>, max = dense<10> : tensor<1xi32>>})

// -----

func.func @func_metadata_conversion(%arg0: memref<?xi32> {plan.result_slot = 0 : i32}) attributes {
  plan.shape_func = @shape_func
} {
  return
}

func.func @shape_func(%arg0: memref<2xi32>, %arg1: memref<2xi32> {plan.result_slot = 0 : i32}) {
  return
}

// CHECK-LABEL: func.func @func_metadata_conversion
//  CHECK-SAME: (%{{.+}}: memref<?xi32> {executor.result_slot = 0 : i32}) attributes {executor.shape_func = @shape_func}
//       CHECK: func.func @shape_func(%{{.+}}: memref<2xi32>, %{{.+}}: memref<2xi32> {executor.result_slot = 0 : i32})

// -----

func.func @nvtx_push_single(%arg0: memref<2x4xf32, #plan.memory_space<device>>) -> (memref<2x4xf32, #plan.memory_space<device>>, i64) {
  %0, %1 = plan.nvtx_push {name = "attn", color = -16711936 : i32} %arg0 : (memref<2x4xf32, #plan.memory_space<device>>) -> (memref<2x4xf32, #plan.memory_space<device>>, i64)
  return %0, %1 : memref<2x4xf32, #plan.memory_space<device>>, i64
}

// CHECK-DAG: executor.func private @__nvtx_push(!executor.str_literal, i32, ...) -> i64
// CHECK-LABEL: @nvtx_push_single
//  CHECK-SAME: (%[[arg0:.+]]: memref<2x4xf32, #executor.memory_type<device>>) -> (memref<2x4xf32, #executor.memory_type<device>>, i64)
//       CHECK:   %[[str:.+]] = executor.str_literal "attn"
//       CHECK:   %[[color:.+]] = executor.constant -16711936 : i32
//       CHECK:   %[[range_id:.+]] = executor.call @__nvtx_push(%[[str]], %[[color]]) : (!executor.str_literal, i32) -> i64
//       CHECK:   return %[[arg0]], %[[range_id]] : memref<2x4xf32, #executor.memory_type<device>>, i64

// -----

func.func @nvtx_pop_single(%arg0: memref<2x4xf32, #plan.memory_space<device>>, %arg1: i64) -> memref<2x4xf32, #plan.memory_space<device>> {
  %0 = plan.nvtx_pop %arg0, %arg1 : (memref<2x4xf32, #plan.memory_space<device>>, i64) -> memref<2x4xf32, #plan.memory_space<device>>
  return %0 : memref<2x4xf32, #plan.memory_space<device>>
}

// CHECK-DAG: executor.func private @__nvtx_pop(i64, ...)
// CHECK-LABEL: @nvtx_pop_single
//  CHECK-SAME: (%[[arg0:.+]]: memref<2x4xf32, #executor.memory_type<device>>, %[[arg1:.+]]: i64) -> memref<2x4xf32, #executor.memory_type<device>>
//       CHECK:   executor.call @__nvtx_pop(%[[arg1]]) : (i64) -> ()
//       CHECK:   return %[[arg0]] : memref<2x4xf32, #executor.memory_type<device>>

// -----

func.func @nvtx_push_variadic(
    %arg0: memref<2x4xf32, #plan.memory_space<device>>,
    %arg1: memref<3x4xf32, #plan.memory_space<device>>)
    -> (memref<2x4xf32, #plan.memory_space<device>>,
        memref<3x4xf32, #plan.memory_space<device>>, i64) {
  %0, %1, %2 = plan.nvtx_push {name = "multi_input", color = -16711936 : i32} %arg0, %arg1 : (memref<2x4xf32, #plan.memory_space<device>>, memref<3x4xf32, #plan.memory_space<device>>) -> (memref<2x4xf32, #plan.memory_space<device>>, memref<3x4xf32, #plan.memory_space<device>>, i64)
  return %0, %1, %2 : memref<2x4xf32, #plan.memory_space<device>>, memref<3x4xf32, #plan.memory_space<device>>, i64
}

// CHECK-DAG: executor.func private @__nvtx_push(!executor.str_literal, i32, ...) -> i64
// CHECK-LABEL: @nvtx_push_variadic
//  CHECK-SAME: (%[[arg0:.+]]: memref<2x4xf32, #executor.memory_type<device>>, %[[arg1:.+]]: memref<3x4xf32, #executor.memory_type<device>>) -> (memref<2x4xf32, #executor.memory_type<device>>, memref<3x4xf32, #executor.memory_type<device>>, i64)
//       CHECK:   %[[str:.+]] = executor.str_literal "multi_input"
//       CHECK:   %[[color:.+]] = executor.constant -16711936 : i32
//       CHECK:   %[[range_id:.+]] = executor.call @__nvtx_push(%[[str]], %[[color]]) : (!executor.str_literal, i32) -> i64
//       CHECK:   return %[[arg0]], %[[arg1]], %[[range_id]] : memref<2x4xf32, #executor.memory_type<device>>, memref<3x4xf32, #executor.memory_type<device>>, i64


// -----

// Verifies pass-through: push's input feeds directly to the add, and pop's input feeds directly to return.
func.func @nvtx_push_pop_passthrough(
    %arg0: memref<2x4xf32, #plan.memory_space<device>>)
    -> memref<2x4xf32, #plan.memory_space<device>> {
  %0, %1 = plan.nvtx_push {name = "compute", color = 255 : i32} %arg0 : (memref<2x4xf32, #plan.memory_space<device>>) -> (memref<2x4xf32, #plan.memory_space<device>>, i64)
  %2 = plan.nvtx_pop %0, %1 : (memref<2x4xf32, #plan.memory_space<device>>, i64) -> memref<2x4xf32, #plan.memory_space<device>>
  return %2 : memref<2x4xf32, #plan.memory_space<device>>
}

// CHECK-DAG: executor.func private @__nvtx_pop(i64, ...)
// CHECK-DAG: executor.func private @__nvtx_push(!executor.str_literal, i32, ...) -> i64
// CHECK-LABEL: @nvtx_push_pop_passthrough
//  CHECK-SAME: (%[[arg0:.+]]: memref<2x4xf32, #executor.memory_type<device>>) -> memref<2x4xf32, #executor.memory_type<device>>
//       CHECK:   %[[str:.+]] = executor.str_literal "compute"
//       CHECK:   %[[color:.+]] = executor.constant 255 : i32
//       CHECK:   %[[range_id:.+]] = executor.call @__nvtx_push(%[[str]], %[[color]]) : (!executor.str_literal, i32) -> i64
//       CHECK:   executor.call @__nvtx_pop(%[[range_id]]) : (i64) -> ()
//       CHECK:   return %[[arg0]] : memref<2x4xf32, #executor.memory_type<device>>
