// RUN: mlir-tensorrt-opt %s -split-input-file \
// RUN: -pass-pipeline="builtin.module(plan-assign-memory-spaces,func.func(plan-optimize-memory-spaces))" \
// RUN: | FileCheck %s

func.func private @cond() -> i1

// CHECK-LABEL: func.func @scf_while_loop_2
// CHECK: scf.while {{.*}}tensor<1xf32, #plan.memory_space<host>>) -> tensor<1xf32, #plan.memory_space<host>>
// CHECK-NOT: #plan.memory_space<device>
func.func @scf_while_loop_2(%arg0: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %1 = tensor.from_elements %arg0  : tensor<1xf32>
  %2 = scf.while (%arg1 = %1) : (tensor<1xf32>) -> tensor<1xf32> {
    %cond = func.call @cond() : () -> i1
    %e = tensor.extract %arg1[%c0] : tensor<1xf32>
    %f = arith.addf %e, %e : f32
    %3 = tensor.from_elements %f : tensor<1xf32>
    scf.condition(%cond) %3 : tensor<1xf32>
  } do {
  ^bb0(%arg1: tensor<1xf32>):
    %extract = tensor.extract %arg1[%c0] : tensor<1xf32>
    %3 = arith.addf %extract, %extract : f32
    %4 = tensor.from_elements %3 : tensor<1xf32>
    scf.yield %4 : tensor<1xf32>
  }
  %3 = tensor.extract %2[%c0] : tensor<1xf32>
  return %3 : f32
}

// -----

// CHECK-LABEL: func.func @arith_constant
// CHECK: arith.constant {{.*}} : tensor<2xf32, #plan.memory_space<device>>
// CHECK: arith.constant {{.*}} : tensor<2xf32, #plan.memory_space<device>>
func.func @arith_constant() -> (tensor<2xf32>, tensor<2xf32>) {
  %0 = arith.constant dense<[0.1, 0.2]> : tensor<2xf32>
  %1 = arith.constant dense_resource<__elided__> : tensor<2xf32>
  return %0, %1 : tensor<2xf32>, tensor<2xf32>
}

// -----

// CHECK-LABEL: module @nested_module
// CHECK-NOT: #plan.memory_space
module @outer {
module @nested_module {
  func.func @nested_func() -> tensor<2xf32> {
    %0 = arith.constant dense<[0.1, 0.2]> : tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}
}

// -----

// CHECK-LABEL: func.func @existing_constraint_1
// CHECK: tensor.extract {{.*}}<host>
func.func @existing_constraint_1(%arg0: tensor<2xf32, #plan.memory_space<host>>) -> f32 {
  %c0 = arith.constant 0 : index
  %0 = tensor.extract %arg0[%c0] : tensor<2xf32, #plan.memory_space<host>>
  return %0 : f32
}

// -----

// CHECK-LABEL: func.func @existing_constraint_2
// CHECK-NOT: tensor.cast
// CHECK: tensor.extract {{.*}}<host>
func.func @existing_constraint_2(%arg0: tensor<2xf32, #plan.memory_space<host>>) -> f32 {
  %c0 = arith.constant 0 : index
  %1 = tensor.cast %arg0 : tensor<2xf32, #plan.memory_space<host>> to tensor<2xf32>
  %0 = tensor.extract %1[%c0] : tensor<2xf32>
  return %0 : f32
}

// -----

// CHECK-LABEL: func.func @host_func
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2xf32, #plan.memory_space<host>>, %[[arg1:.+]]: tensor<2xf32, #plan.memory_space<host>>)
//  CHECK-SAME: -> tensor<2xf32, #plan.memory_space<host>>
func.func @host_func(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32>
    attributes {plan.cluster_kind = #plan.host_cluster<benefit = 1>} {
  // CHECK: %[[v0:.+]] = arith.addf %[[arg0]], %[[arg1]] : tensor<2xf32, #plan.memory_space<host>>
  %0 = arith.addf %arg0, %arg1 : tensor<2xf32>
  // CHECK: return %[[v0]]
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func.func @default_func
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2xf32, #plan.memory_space<device>>, %[[arg1:.+]]: tensor<2xf32, #plan.memory_space<host>>
//  CHECK-SAME:  -> (tensor<2xf32, #plan.memory_space<device>>, tensor<2xf32, #plan.memory_space<host>> {plan.memory_space = #plan.memory_space<host>})
func.func @default_func(%arg0: tensor<2xf32>, %arg1: tensor<2xf32> {plan.memory_space = #plan.memory_space<host>}) -> (tensor<2xf32>, tensor<2xf32> {plan.memory_space = #plan.memory_space<host>}) {
  // CHECK-DAG: %[[cast:.+]] = tensor.cast %[[arg0]]
  // CHECK-DAG: %[[v0:.+]] = call @host_func(%[[cast]], %[[arg1]]) :
  %0 = func.call @host_func(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  // CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[cast_0:.+]] = tensor.cast %[[v0]] : tensor<2xf32, #plan.memory_space<host>> to tensor<2xf32, #plan.memory_space<device>>
  // CHECK-DAG: %[[extracted:.+]] = tensor.extract %[[v0]][%[[c0]]] : tensor<2xf32, #plan.memory_space<host>>
  // CHECK-DAG: %[[inserted:.+]] = tensor.insert %[[extracted]] into %[[v0]][%[[c0]]]
  // CHECK-DAG: return %[[cast_0]], %[[inserted]]
  %c0 = arith.constant 0 : index
  %1 = tensor.extract %0[%c0] : tensor<2xf32>
  %2 = tensor.insert %1 into %0[%c0] : tensor<2xf32>
  return %0, %2 : tensor<2xf32>, tensor<2xf32>
}

// -----


// CHECK-LABEL: module @test_decl
// CHECK-LABEL: func.func private @decl(tensor<{{.*}}device>>, tensor<{{.*}}host>> {plan.memory_space = #plan.memory_space<host>}) -> (tensor<{{.*}}host>> {plan.memory_space = #plan.memory_space<host>}, tensor<{{.*}}device>>)

module @test_decl {

func.func private @decl(tensor<2xf32>, tensor<2xf32> {plan.memory_space = #plan.memory_space<host>})
                        -> (tensor<2xf32> {plan.memory_space = #plan.memory_space<host>}, tensor<2xf32>)

// CHECK-LABEL: func.func @caller
// CHECK-SAME: (%[[arg0:.+]]: tensor<2xf32, #plan.memory_space<device>>, %[[arg1:.+]]: tensor<2xf32, #plan.memory_space<device>>
func.func @caller(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK-DAG: %[[cast:.+]] = tensor.cast %[[arg1]] : tensor<2xf32, #plan.memory_space<device>> to tensor<2xf32, #plan.memory_space<host>>
  // CHECK-DAG: %[[v0:.+]]:2 = call @decl(%[[arg0]], %[[cast]])
  // CHECK-DAG: %[[v1:.+]] = tensor.cast %[[v0]]#0
  // CHECK-DAG: %[[v2:.+]] = arith.addf %[[v1]], %[[v0]]#1
  // CHECK-DAG: return %[[v2]]
  %0:2 = func.call @decl(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)
  %1 = arith.addf %0#0, %0#1 : tensor<2xf32>
  return %1 : tensor<2xf32>
}

}

// -----

// CHECK-LABEL: func.func @multiple_blocks
func.func @multiple_blocks(
      %c: i1,
      %th: tensor<10xf32> {plan.memory_space = #plan.memory_space<host>})
     -> (tensor<5xf32> {plan.memory_space = #plan.memory_space<device>}) {
  // CHECK: %[[cast:.+]] = tensor.cast %[[arg1]] {{.*}}device>>
  // CHECK: %[[v0:.+]] = tensor.empty() {{.*}}device>>
  %td = tensor.empty() : tensor<10xf32>
  // CHECK: cf.cond_br
  cf.cond_br %c, ^bb1, ^bb2
// CHECK: ^bb1
^bb1:
  // CHECK-DAG: %[[es0:.+]] = tensor.extract_slice %[[v0]]
  %0 = tensor.extract_slice %td[0][5][2] : tensor<10xf32> to tensor<5xf32>
  // CHECK-DAG: return %[[es0]] {{.*}}device>>
  return %0 : tensor<5xf32>
// CHECK: ^bb2
^bb2:
  // CHECK-DAG: %[[es0:.+]] = tensor.extract_slice %[[cast]]
  %1 = tensor.extract_slice %th[1][5][1] : tensor<10xf32> to tensor<5xf32>
  // CHECK-DAG: return %[[es0]] {{.*}}device>>
  return %1 : tensor<5xf32>
}

// -----

// Test that the `plan.memory_space` attribute on a function is respected
// but can be overriden by other constraints.


func.func @function_level_override(
  %arg0: tensor<2xi32>,
  %arg1: tensor<2xi32> {plan.memory_space = #plan.memory_space<device>}
) ->
  (tensor<2xindex, #plan.memory_space<host_pinned>>,
   tensor<2xi32>,
   tensor<2xf32> {plan.memory_space = #plan.memory_space<device>})
     attributes {plan.memory_space = #plan.memory_space<host>} {
  %cst = arith.constant dense<0> : tensor<2xindex>
  %cast = tensor.cast %cst : tensor<2xindex> to tensor<2xindex, #plan.memory_space<host_pinned>>
  %cst1 = arith.constant dense<1> : tensor<2xi32>
  %cst2 = arith.constant dense<2.0> : tensor<2xf32>
  return %cast, %cst1, %cst2
    : tensor<2xindex, #plan.memory_space<host_pinned>>, tensor<2xi32>, tensor<2xf32>
}

// CHECK-LABEL: func.func @function_level_override
// CHECK-SAME: (%{{.+}}: tensor<2xi32, #plan.memory_space<host>>,
// CHECK-SAME:  %{{.+}}: tensor<2xi32, #plan.memory_space<device>>
// CHECK-SAME:  -> (tensor<2xindex, #plan.memory_space<host_pinned>>,
// CHECK-SAME:   tensor<2xi32, #plan.memory_space<host>>,
// CHECK-SAME:   tensor<2xf32, #plan.memory_space<device>>

// -----

func.func @tensor_reshape(
  %arg0: tensor<?xi32>,
  %arg1: i32,
  %arg2: i32
) -> tensor<?x?xi32> {
  %0 = tensor.from_elements %arg1, %arg2 : tensor<2xi32>
  %1 = tensor.reshape %arg0(%0) : (tensor<?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// CHECK-LABEL: func.func @tensor_reshape
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xi32, #plan.memory_space<device>>, %[[arg1:.+]]: i32, %[[arg2:.+]]: i32)
//   CHECK-DAG:     %[[from_elements:.+]] = tensor.from_elements %[[arg1]], %[[arg2]] : tensor<2xi32, #plan.memory_space<host>>
//   CHECK-DAG:     %[[reshape:.+]] = tensor.reshape %[[arg0]](%[[from_elements]])
//   CHECK-DAG:     return %[[reshape]]

// -----

func.func @alloc_tensor() -> tensor<2x128xf32> {
  %0 = bufferization.alloc_tensor() {
    memory_space = #plan.memory_space<host>
  } : tensor<2x128xf32>
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.0 : f32
  %1 = tensor.insert %cst into %0[%c0, %c0] : tensor<2x128xf32>
  return %1 : tensor<2x128xf32>
}

// CHECK-LABEL: func.func @alloc_tensor
//  CHECK-SAME: () -> tensor<2x128xf32, #plan.memory_space<device>>
//   CHECK-DAG:   %[[cst:.+]] = arith.constant 1.000000e+00 : f32
//   CHECK-DAG:   %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[v1:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<host>} : tensor<2x128xf32, #plan.memory_space<host>>
//   CHECK-DAG:   %[[inserted:.+]] = tensor.insert %[[cst]] into %[[v1]][%[[c0]], %[[c0]]]
//   CHECK-DAG:   %[[cast:.+]] = tensor.cast %[[inserted]] : tensor<2x128xf32, #plan.memory_space<host>> to tensor<2x128xf32, #plan.memory_space<device>>
//   CHECK-DAG:   return %[[cast]]

// -----

func.func @alloc_tensor_2(%arg0: tensor<2x128xf32>) -> tensor<2x128xf32> {
  %0 = bufferization.alloc_tensor() copy(%arg0) {
    memory_space = #plan.memory_space<host>
  } : tensor<2x128xf32>
  return %0 : tensor<2x128xf32>
}

// CHECK-LABEL: func.func @alloc_tensor_2
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x128xf32, #plan.memory_space<device>>) -> tensor<2x128xf32, #plan.memory_space<device>> {
//       CHECK:     return %[[arg0]]

// -----

func.func @large_constant() -> (tensor<1024xindex> {plan.memory_space = #plan.memory_space<host>}) {
  %0 = arith.constant dense<1> : tensor<1024xindex>
  return %0 : tensor<1024xindex>
}

// CHECK-LABEL: func.func @large_constant
//  CHECK-SAME:  -> (tensor<1024xindex, #plan.memory_space<host>>
//       CHECK:     %[[cst:.+]] = arith.constant dense<1> : tensor<1024xindex, #plan.memory_space<host>>
//       CHECK:     return %[[cst]] : tensor<1024xindex, #plan.memory_space<host>>


// -----

func.func @small_host_tensor_constant(%arg0: tensor<?x?xf32>) -> (tensor<?x?x?x?xf32>) {
  %0 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xindex>
  %1 = tensor.reshape %arg0 (%0) : (tensor<?x?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: func.func @small_host_tensor_constant
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?xf32, #plan.memory_space<device>>) -> tensor<?x?x?x?xf32, #plan.memory_space<device>> {
//   CHECK-DAG:     %[[cst:.+]] = arith.constant dense<[1, 2, 3, 4]> : tensor<4xindex, #plan.memory_space<host>>
//   CHECK-DAG:     %[[reshape:.+]] = tensor.reshape %[[arg0]](%[[cst]]) :
//   CHECK-DAG:     return %[[reshape]] :

// -----

func.func @small_host_and_device_tensor_constant(%arg0: tensor<?x?xf32>) -> (tensor<?x?x?x?xf32>, tensor<4xindex>) {
  %0 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xindex>
  %1 = tensor.reshape %arg0 (%0) : (tensor<?x?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  return %1, %0 : tensor<?x?x?x?xf32>, tensor<4xindex>
}

// CHECK-LABEL: func.func @small_host_and_device_tensor_constant
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?xf32, #plan.memory_space<device>>)
//  CHECK-SAME: -> (tensor<?x?x?x?xf32, #plan.memory_space<device>>, tensor<4xindex, #plan.memory_space<device>>) {
//   CHECK-DAG:     %[[cst:.+]] = arith.constant dense<[1, 2, 3, 4]> : tensor<4xindex, #plan.memory_space<device>>
//   CHECK-DAG:     %[[cst_host:.+]] = arith.constant dense<[1, 2, 3, 4]> : tensor<4xindex, #plan.memory_space<host>>
//   CHECK-DAG:     %[[reshape:.+]] = tensor.reshape %[[arg0]](%[[cst_host]]) :
//   CHECK-DAG:     return %[[reshape]], %[[cst]] :
