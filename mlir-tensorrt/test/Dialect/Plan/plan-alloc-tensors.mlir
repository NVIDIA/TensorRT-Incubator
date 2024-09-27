// RUN: mlir-tensorrt-opt %s -split-input-file -plan-alloc-tensors | FileCheck %s

func.func @tensor_empty() -> tensor<128xf32> {
  %0 = tensor.empty() : tensor<128xf32>
  return %0 : tensor<128xf32>
}

// CHECK-LABEL: @tensor_empty
//  CHECK-SAME: (%[[arg0:.+]]: tensor<128xf32> {plan.result_arg}) -> tensor<128xf32> {
//       CHECK:     %[[v0:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<device>} : tensor<128xf32>
//       CHECK:     %[[v1:.+]] = bufferization.materialize_in_destination %[[v0]] in %[[arg0]] :
//       CHECK:     return %[[v1]] : tensor<128xf32>

// -----

func.func @from_elements(%arg0: f32, %arg1: f32) -> tensor<2xf32> {
  %0 = tensor.from_elements %arg0, %arg1 : tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func.func @from_elements
//  CHECK-SAME: (%[[arg0:.+]]: f32, %[[arg1:.+]]: f32, %[[arg2:.+]]: tensor<2xf32> {plan.result_arg}) -> tensor<2xf32> {
//       CHECK:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[v0:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<host_pinned>} : tensor<2xf32, #plan.memory_space<host_pinned>>
//       CHECK:     %[[inserted:.+]] = tensor.insert %[[arg0]] into %[[v0]][%[[c0]]] : tensor<2xf32, #plan.memory_space<host_pinned>>
//       CHECK:     %[[inserted_0:.+]] = tensor.insert %[[arg1]] into %[[inserted]][%[[c1]]] : tensor<2xf32, #plan.memory_space<host_pinned>>
//       CHECK:     %[[cast:.+]] = tensor.cast %[[arg2]] : tensor<2xf32> to tensor<2xf32, #plan.memory_space<device>>
//       CHECK:     %[[v1:.+]] = bufferization.materialize_in_destination %[[inserted_0]] in %[[cast]] : (tensor<2xf32, #plan.memory_space<host_pinned>>, tensor<2xf32, #plan.memory_space<device>>) -> tensor<2xf32, #plan.memory_space<device>>
//       CHECK:     %[[cast_1:.+]] = tensor.cast %[[v1]] : tensor<2xf32, #plan.memory_space<device>> to tensor<2xf32>
//       CHECK:     return %[[cast_1]] : tensor<2xf32>
// -----

#map = affine_map<(d0)->(d0)>
func.func @test_simple_dps(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %empty = tensor.empty () : tensor<10xf32>
  %0 = linalg.generic {
    iterator_types = ["parallel"],
    indexing_maps = [#map, #map]
  } ins(%arg0: tensor<10xf32>) outs(%empty: tensor<10xf32>) {
    ^bb0(%a: f32, %b: f32):
      %r = arith.negf %a : f32
      linalg.yield %r : f32
  } -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @test_simple_dps
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32> {plan.result_arg}) -> tensor<10xf32>
//   CHECK-NOT: bufferization.alloc_tensor()
//       CHECK: %[[v0:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<10xf32>) outs(%[[arg1]] : tensor<10xf32>)
//       CHECK: return %[[v0]] : tensor<10xf32>

// -----

#map = affine_map<(d0)->(d0)>
func.func @test_two_returns(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
  %empty = tensor.empty () : tensor<10xf32>
  %0, %1 = linalg.generic {
    iterator_types = ["parallel"],
    indexing_maps = [#map, #map, #map, #map]
  } ins(%arg0, %arg1: tensor<10xf32>, tensor<10xf32>) outs(%empty, %empty: tensor<10xf32>, tensor<10xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32, %d: f32):
      %r1 = arith.negf %a : f32
      %r2 = arith.negf %b : f32
      linalg.yield %r1, %r2 : f32, f32
  } -> (tensor<10xf32>, tensor<10xf32>)
  return %0, %1 : tensor<10xf32>, tensor<10xf32>
}

// CHECK-LABEL: @test_two_returns
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32>, %[[arg2:.+]]: tensor<10xf32> {plan.result_arg}, %[[arg3:.+]]: tensor<10xf32> {plan.result_arg}) -> (tensor<10xf32>, tensor<10xf32>)
//   CHECK-NOT: bufferization.alloc_tensor()
//       CHECK: %[[v0:.+]]:2 = linalg.generic {{.*}} ins(%[[arg0]], %[[arg1]] : {{.*}}) outs(%[[arg2]], %[[arg3]] : {{.*}})
//       CHECK: return %[[v0]]#0, %[[v0]]#1 : tensor<10xf32>, tensor<10xf32>

// -----

#map = affine_map<(d0)->(d0)>
module @test_no_dps_return {
  func.func @main(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
    %empty = tensor.empty () : tensor<10xf32>
    %0, %1 = linalg.generic {
      iterator_types = ["parallel"],
      indexing_maps = [#map, #map, #map, #map]
    } ins(%arg0, %arg1: tensor<10xf32>, tensor<10xf32>) outs(%empty, %empty: tensor<10xf32>, tensor<10xf32>) {
      ^bb0(%a: f32, %b: f32, %c: f32, %d: f32):
        %r1 = arith.negf %a : f32
        %r2 = arith.negf %b : f32
        linalg.yield %r1, %r2 : f32, f32
    } -> (tensor<10xf32>, tensor<10xf32>)
    %2 = arith.addf %0, %1 : tensor<10xf32>
    return %2 : tensor<10xf32>
  }
}

// CHECK-LABEL: @main
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32>, %[[arg2:.+]]: tensor<10xf32> {plan.result_arg}) -> tensor<10xf32>
//  CHECK-NEXT: %[[v0:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<device>} : tensor<10xf32>
//  CHECK-NEXT: %[[v1:.+]]:2 = {{.*}} ins(%[[arg0]], %[[arg1]] : {{.*}}) outs(%[[v0]], %[[v0]] : {{.*}})
//       CHECK: %[[v2:.+]] = arith.addf %[[v1]]#0, %[[v1]]#1 : tensor<10xf32>
//  CHECK-NEXT: %[[v3:.+]] = bufferization.materialize_in_destination %[[v2]] in %[[arg2]] :
//  CHECK-NEXT: return %[[v3]] : tensor<10xf32>

// -----

#map = affine_map<(d0)->(d0)>
module @test_one_dps_other_no_dps_return {
  func.func @main(%arg0: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
    %empty = tensor.empty () : tensor<10xf32>
    %0 = linalg.generic {
      iterator_types = ["parallel"],
      indexing_maps = [#map, #map]
    } ins(%arg0: tensor<10xf32>) outs(%empty: tensor<10xf32>) {
      ^bb0(%a: f32, %b: f32):
        %r = arith.negf %a : f32
        linalg.yield %r : f32
    } -> tensor<10xf32>
    %1 = arith.addf %0, %arg0 : tensor<10xf32>
    return %1, %0 : tensor<10xf32>, tensor<10xf32>
  }
}

//  CHECK-LABEL: @main
//   CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32> {plan.result_arg}, %[[arg2:.+]]: tensor<10xf32> {plan.result_arg}) -> (tensor<10xf32>, tensor<10xf32>)
//    CHECK-NOT: bufferization.alloc_tensor()
//        CHECK: %[[v0:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<10xf32>) outs(%[[arg2]] : tensor<10xf32>)
//        CHECK: %[[v1:.+]] = arith.addf %[[v0]], %[[arg0]] : tensor<10xf32>
//   CHECK-NEXT: %[[v2:.+]] = bufferization.materialize_in_destination %[[v1]] in %[[arg1]] :
//   CHECK-NEXT: return %[[v2]], %[[v0]] : tensor<10xf32>, tensor<10xf32>

// -----

#map = affine_map<(d0)->(d0)>
func.func @test_dps_chain(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %empty = tensor.empty () : tensor<10xf32>
  %0 = linalg.generic {
    iterator_types = ["parallel"],
    indexing_maps = [#map, #map]
  } ins(%arg0: tensor<10xf32>) outs(%empty: tensor<10xf32>) {
    ^bb0(%a: f32, %b: f32):
      %r = arith.negf %a : f32
      linalg.yield %r : f32
  } -> tensor<10xf32>
  %2 = linalg.generic {
    iterator_types = ["parallel"],
    indexing_maps = [#map, #map]
  } ins(%arg0: tensor<10xf32>) outs(%0: tensor<10xf32>) {
    ^bb0(%a: f32, %b: f32):
      %r = arith.negf %a : f32
      linalg.yield %r : f32
  } -> tensor<10xf32>
  return %2 : tensor<10xf32>
}

// CHECK-LABEL: @test_dps_chain
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32> {plan.result_arg}) -> tensor<10xf32>
//   CHECK-NOT: bufferization.alloc_tensor()
//       CHECK: %[[v0:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<10xf32>) outs(%[[arg1]] : tensor<10xf32>)
//       CHECK: %[[v1:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<10xf32>) outs(%[[v0]] : tensor<10xf32>)
//       CHECK: return %[[v1]] : tensor<10xf32>

// -----

#map = affine_map<(d0)->(d0)>
func.func @test_repeat_returns(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>) {
  %empty = tensor.empty () : tensor<10xf32>
  %0, %1 = linalg.generic {
    iterator_types = ["parallel"],
    indexing_maps = [#map, #map, #map, #map]
  } ins(%arg0, %arg1: tensor<10xf32>, tensor<10xf32>) outs(%empty, %empty: tensor<10xf32>, tensor<10xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32, %d: f32):
      %r1 = arith.negf %a : f32
      %r2 = arith.negf %b : f32
      linalg.yield %r1, %r2 : f32, f32
  } -> (tensor<10xf32>, tensor<10xf32>)
  %2 = arith.addf %0, %arg0 : tensor<10xf32>
  return %0, %1, %0, %1, %2 : tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>
}

// CHECK-LABEL: @test_repeat_returns
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32>, %[[arg2:.+]]: tensor<10xf32> {plan.result_arg}, %[[arg3:.+]]: tensor<10xf32> {plan.result_arg}, %[[arg4:.+]]: tensor<10xf32> {plan.result_arg}, %[[arg5:.+]]: tensor<10xf32> {plan.result_arg}, %[[arg6:.+]]: tensor<10xf32> {plan.result_arg})
//   CHECK-NOT: bufferization.alloc_tensor()
//       CHECK: %[[v0:.+]]:2 = linalg.generic {{.*}} ins(%[[arg0]], %[[arg1]] : {{.*}}) outs(%[[arg2]], %[[arg3]] : {{.*}})
//       CHECK: %[[v1:.+]] = arith.addf %[[v0]]#0, %[[arg0]] : tensor<10xf32>
//  CHECK-NEXT: %[[v2:.+]] = bufferization.materialize_in_destination %[[v0]]#0 in %[[arg4]] :
//  CHECK-NEXT: %[[v3:.+]] = bufferization.materialize_in_destination %[[v0]]#1 in %[[arg5]] :
//  CHECK-NEXT: %[[v4:.+]] = bufferization.materialize_in_destination %[[v1]] in %[[arg6]] :
//  CHECK-NEXT: return %[[v0]]#0, %[[v0]]#1, %[[v2]], %[[v3]], %[[v4]] : tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>

// -----

#map = affine_map<(d0)->(d0)>
func.func @test_dps_chain_repeat(%arg0: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>) {
  %empty = tensor.empty () : tensor<10xf32>
  %0 = linalg.generic {
    iterator_types = ["parallel"],
    indexing_maps = [#map, #map]
  } ins(%arg0: tensor<10xf32>) outs(%empty: tensor<10xf32>) {
    ^bb0(%a: f32, %b: f32):
      %r = arith.negf %a : f32
      linalg.yield %r : f32
  } -> tensor<10xf32>
  %2 = linalg.generic {
    iterator_types = ["parallel"],
    indexing_maps = [#map, #map]
  } ins(%arg0: tensor<10xf32>) outs(%0: tensor<10xf32>) {
    ^bb0(%a: f32, %b: f32):
      %r = arith.negf %a : f32
      linalg.yield %r : f32
  } -> tensor<10xf32>
  return %2, %0, %0 : tensor<10xf32>, tensor<10xf32>, tensor<10xf32>
}

// CHECK-LABEL: @test_dps_chain_repeat
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32> {plan.result_arg}, %[[arg2:.+]]: tensor<10xf32> {plan.result_arg}, %[[arg3:.+]]: tensor<10xf32> {plan.result_arg}) -> (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>)
//   CHECK-NOT: bufferization.alloc_tensor()
//       CHECK: %[[v0:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<10xf32>) outs(%[[arg1]] : tensor<10xf32>)
//       CHECK: %[[v1:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<10xf32>) outs(%[[v0]] : tensor<10xf32>)
//       CHECK: %[[v2:.+]] = bufferization.materialize_in_destination %[[v0]] in %[[arg2]] :
//  CHECK-NEXT: %[[v3:.+]] = bufferization.materialize_in_destination %[[v0]] in %[[arg3]] :
//  CHECK-NEXT: return %[[v1]], %[[v2]], %[[v3]] : tensor<10xf32>, tensor<10xf32>, tensor<10xf32>

// -----

func.func @test_return_arg(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  return %arg0: tensor<10xf32>
}

// CHECK-LABEL: @test_return_arg
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32> {plan.result_arg})
//       CHECK:     %[[v0:.+]] = bufferization.materialize_in_destination %[[arg0]] in %[[arg1]]
//       CHECK:     return %[[v0]]

// -----

func.func @test_already_dps(%arg0: tensor<10xf32>, %arg1: tensor<10xf32> {plan.result_arg}, %arg2: tensor<10xf32> {plan.result_arg}) -> (tensor<10xf32>, tensor<10xf32>) {
  %0 = bufferization.materialize_in_destination %arg0 in %arg1 : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  %1 = bufferization.materialize_in_destination %arg0 in %arg2 : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %0, %1 : tensor<10xf32>, tensor<10xf32>
}

// CHECK-LABEL: @test_already_dps
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32> {plan.result_arg}, %[[arg2:.+]]: tensor<10xf32> {plan.result_arg})
//       CHECK:     %[[v0:.+]] = bufferization.materialize_in_destination %[[arg0]] in %[[arg1]]
//       CHECK:     %[[v1:.+]] = bufferization.materialize_in_destination %[[arg0]] in %[[arg2]]
//       CHECK:     return %[[v0]], %[[v1]] : tensor<10xf32>, tensor<10xf32>


// -----

func.func @test_loop_region_dps_rewrite_while(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %c0 = arith.constant 0 : index
  %c0f = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<10xf32>
  %r = scf.while(%arg1 = %arg0) : (tensor<10xf32>) -> (tensor<10xf32>) {
    %v0 = tensor.extract %arg1[%c0] : tensor<10xf32>
    %cond = arith.cmpf ogt, %v0, %c0f : f32
    scf.condition(%cond) %arg1 : tensor<10xf32>
  } do {
  ^bb0(%arg2: tensor<10xf32>):
    %1 = linalg.map {math.exp}
      ins(%arg2 : tensor<10xf32>)
      outs(%0 : tensor<10xf32>)
    scf.yield %1 : tensor<10xf32>
  }
  return %r : tensor<10xf32>
}

// CHECK-LABEL: @test_loop_region_dps_rewrite_while
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32> {plan.result_arg}) -> tensor<10xf32> {
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 :
//   CHECK-DAG:     %[[cst:.+]] = arith.constant 0.000000e+00
//       CHECK:     %[[v0:.+]] = scf.while (%[[argB:.+]] = %[[arg0]])
//       CHECK:       %[[v2:.+]] = bufferization.alloc_tensor() copy(%[[argB]])
//       CHECK:       %[[extracted:.+]] = tensor.extract %[[v2]][%[[c0]]] : tensor<10xf32>
//       CHECK:       %[[v3:.+]] = arith.cmpf ogt, %[[extracted]], %[[cst]] : f32
//       CHECK:       scf.condition(%[[v3]]) %[[argB]] : tensor<10xf32>
//       CHECK:     } do {
//       CHECK:     ^bb0(%[[arg3:.+]]: tensor<10xf32>):
//       CHECK:       %[[mapped:.+]] = linalg.map { math.exp } ins(%[[arg3]] : tensor<10xf32>) outs(%[[arg3]] :
//       CHECK:       scf.yield %[[mapped]] :
//       CHECK:     %[[v1:.+]] = bufferization.materialize_in_destination %[[v0]] in %[[arg1]] :
//       CHECK:     return %[[v1]] :


// -----

func.func @test_loop_region_dps_rewrite_for(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.empty() : tensor<10xf32>
  %r = scf.for %i = %c0 to %c10 step %c1 iter_args(%arg1 = %arg0) -> tensor<10xf32> {
    %1 = linalg.map {math.exp}
      ins(%arg1 : tensor<10xf32>)
      outs(%0 : tensor<10xf32>)
    scf.yield %1 : tensor<10xf32>
  }
  return %r : tensor<10xf32>
}

// CHECK-LABEL: @test_loop_region_dps_rewrite_for
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32> {plan.result_arg})
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[c10:.+]] = arith.constant 10 : index
//       CHECK:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[v0:.+]] = scf.for %[[arg2:.+]] = %[[c0]] to %[[c10]] step %[[c1]] iter_args(%[[arg3:.+]] = %[[arg0]])
//       CHECK:       %[[mapped:.+]] = linalg.map { math.exp } ins(%[[arg3]] : tensor<10xf32>) outs(%[[arg3]] : tensor<10xf32>)
//       CHECK:       scf.yield %[[mapped]] : tensor<10xf32>
//       CHECK:     %[[v1:.+]] = bufferization.materialize_in_destination %[[v0]] in %[[arg1]]
//       CHECK:     return %[[v1]]

// -----

func.func @alloc_tensors_from_elements(%arg0: i32) -> (tensor<1xi32> {tensorrt.host_tensor}, tensor<1xi32>) {
  %0 = tensor.from_elements %arg0 : tensor<1xi32>
  %1 = tensor.from_elements %arg0 : tensor<1xi32>
  return %0, %1 : tensor<1xi32>, tensor<1xi32>
}

// CHECK-LABEL: func.func @alloc_tensors_from_elements
//  CHECK-SAME: (%[[arg0:.+]]: i32, %[[arg1:.+]]: tensor<1xi32> {plan.result_arg}, %[[arg2:.+]]: tensor<1xi32> {plan.result_arg}) -> (tensor<1xi32> {tensorrt.host_tensor}, tensor<1xi32>) {
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[v0:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<host_pinned>}
//       CHECK:     %[[inserted:.+]] = tensor.insert %[[arg0]] into %[[v0]][%[[c0]]] :
//       CHECK:     %[[cast:.+]] = tensor.cast %[[inserted]] : tensor<1xi32, #plan.memory_space<host_pinned>> to tensor<1xi32>
//       CHECK:     %[[v1:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<host_pinned>}
//       CHECK:     %[[inserted_0:.+]] = tensor.insert %[[arg0]] into %[[v1]][%[[c0]]]
//       CHECK:     %[[cast_1:.+]] = tensor.cast %[[arg2]] : tensor<1xi32> to tensor<1xi32, #plan.memory_space<device>>
//       CHECK:     %[[v2:.+]] = bufferization.materialize_in_destination %[[inserted_0]] in %[[cast_1]] :
//       CHECK:     %[[cast_2:.+]] = tensor.cast %[[v2]] : tensor<1xi32, #plan.memory_space<device>> to tensor<1xi32>
//       CHECK:     %[[v3:.+]] = bufferization.materialize_in_destination %[[cast]] in %[[arg1]]
//       CHECK:     return %[[v3]], %[[cast_2]] : tensor<1xi32>, tensor<1xi32>

// -----

func.func @alloc_tensors_from_elements(%arg0: i32) -> (tensor<1xi32> {tensorrt.host_tensor}, tensor<1xi32>) {
  %0 = tensor.from_elements %arg0 : tensor<1xi32>
  return %0, %0 : tensor<1xi32>, tensor<1xi32>
}

// CHECK-LABEL: func.func @alloc_tensors_from_elements
//  CHECK-SAME: (%[[arg0:.+]]: i32, %[[arg1:.+]]: tensor<1xi32> {plan.result_arg}, %[[arg2:.+]]: tensor<1xi32> {plan.result_arg}) -> (tensor<1xi32> {tensorrt.host_tensor}, tensor<1xi32>) {
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[v0:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<host_pinned>} : tensor<1xi32, #plan.memory_space<host_pinned>>
//       CHECK:     %[[inserted:.+]] = tensor.insert %[[arg0]] into %[[v0]][%[[c0]]]
//       CHECK:     %[[cast:.+]] = tensor.cast %[[inserted]] : tensor<1xi32, #plan.memory_space<host_pinned>> to tensor<1xi32>
//       CHECK:     %[[cast_0:.+]] = tensor.cast %[[arg2]] : tensor<1xi32> to tensor<1xi32, #plan.memory_space<device>>
//       CHECK:     %[[v1:.+]] = bufferization.materialize_in_destination %[[inserted]] in %[[cast_0]]
//       CHECK:     %[[cast_1:.+]] = tensor.cast %[[v1]] : tensor<1xi32, #plan.memory_space<device>> to tensor<1xi32>
//       CHECK:     %[[v2:.+]] = bufferization.materialize_in_destination %[[cast]] in %[[arg1]] : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
//       CHECK:     return %[[v2]], %[[cast_1]] : tensor<1xi32>, tensor<1xi32>
//       CHECK:   }

// -----

func.func @small_host_tensor_constant(%arg0: tensor<?x?xf32>) -> (tensor<?x?x?x?xf32>) {
  %0 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xindex>
  %1 = tensor.reshape %arg0 (%0) : (tensor<?x?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: @small_host_tensor_constant
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?xf32>, %[[arg1:.+]]: tensor<?x?x?x?xf32> {plan.result_arg}) -> tensor<?x?x?x?xf32> {
//       CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK-DAG:     %[[c2:.+]] = arith.constant 2 : index
//       CHECK-DAG:     %[[c3:.+]] = arith.constant 3 : index
//       CHECK-DAG:     %[[c4:.+]] = arith.constant 4 : index
//       CHECK:     %[[v0:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<host>} : tensor<4xindex>
//       CHECK:     %[[inserted:.+]] = tensor.insert %[[c1]] into %[[v0]][%[[c0]]] : tensor<4xindex>
//       CHECK:     %[[inserted_0:.+]] = tensor.insert %[[c2]] into %[[inserted]][%[[c1]]] : tensor<4xindex>
//       CHECK:     %[[inserted_1:.+]] = tensor.insert %[[c3]] into %[[inserted_0]][%[[c2]]] : tensor<4xindex>
//       CHECK:     %[[inserted_2:.+]] = tensor.insert %[[c4]] into %[[inserted_1]][%[[c3]]] : tensor<4xindex>
//       CHECK:     %[[reshape:.+]] = tensor.reshape %[[arg0]](%[[inserted_2]]) : (tensor<?x?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
//       CHECK:     %[[v1:.+]] = bufferization.materialize_in_destination %[[reshape]] in %[[arg1]] : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
//       CHECK:     return %[[v1]] : tensor<?x?x?x?xf32>

// -----

func.func @small_host_and_device_tensor_constant(%arg0: tensor<?x?xf32>) -> (tensor<?x?x?x?xf32>, tensor<4xindex>) {
  %0 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xindex>
  %1 = tensor.reshape %arg0 (%0) : (tensor<?x?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  return %1, %0 : tensor<?x?x?x?xf32>, tensor<4xindex>
}

// CHECK-LABEL: @small_host_and_device_tensor_constant
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?xf32>, %[[arg1:.+]]: tensor<?x?x?x?xf32> {plan.result_arg}, %[[arg2:.+]]: tensor<4xindex> {plan.result_arg}) -> (tensor<?x?x?x?xf32>, tensor<4xindex>) {
//   CHECK-DAG:     %[[cst:.+]] = arith.constant dense<[1, 2, 3, 4]> : tensor<4xindex>
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c2:.+]] = arith.constant 2 : index
//   CHECK-DAG:     %[[c3:.+]] = arith.constant 3 : index
//   CHECK-DAG:     %[[c4:.+]] = arith.constant 4 : index
//   CHECK-DAG:     %[[v0:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<host>} : tensor<4xindex>
//       CHECK:     %[[inserted:.+]] = tensor.insert %[[c1]] into %[[v0]][%[[c0]]] : tensor<4xindex>
//       CHECK:     %[[inserted_0:.+]] = tensor.insert %[[c2]] into %[[inserted]][%[[c1]]] : tensor<4xindex>
//       CHECK:     %[[inserted_1:.+]] = tensor.insert %[[c3]] into %[[inserted_0]][%[[c2]]] : tensor<4xindex>
//       CHECK:     %[[inserted_2:.+]] = tensor.insert %[[c4]] into %[[inserted_1]][%[[c3]]] : tensor<4xindex>
//       CHECK:     %[[reshape:.+]] = tensor.reshape %[[arg0]](%[[inserted_2]]) :
//   CHECK-DAG:     %[[v1:.+]] = bufferization.materialize_in_destination %[[reshape]] in %[[arg1]] :
//   CHECK-DAG:     %[[v2:.+]] = bufferization.materialize_in_destination %[[cst]] in %[[arg2]] :
//       CHECK:     return %[[v1]], %[[v2]] :

// -----

func.func @big_host_tensor_constant(%arg0: tensor<?x?xf32>) -> (tensor<1024xindex> {tensorrt.host_tensor}) {
  %0 = arith.constant dense<1> : tensor<1024xindex>
  return %0 : tensor<1024xindex>
}

// CHECK-LABEL: @big_host_tensor_constant
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?xf32>, %[[arg1:.+]]: tensor<1024xindex> {plan.result_arg}) -> (tensor<1024xindex> {tensorrt.host_tensor})
//       CHECK:     %[[cst:.+]] = arith.constant dense<1> : tensor<1024xindex>
//       CHECK:     %[[v0:.+]] = bufferization.alloc_tensor() copy(%[[cst]]) {memory_space = #plan.memory_space<host_pinned>}
//       CHECK:     %[[v1:.+]] = bufferization.materialize_in_destination %[[v0]] in %[[arg1]]
//       CHECK:     return %[[v1]] : tensor<1024xindex>

// -----

func.func @device_extract(%arg0: tensor<128xi1>, %arg1: index) -> i1 {
  %1 = tensor.extract %arg0[%arg1] : tensor<128xi1>
  return %1 : i1
}

// CHECK-LABEL: @device_extract
//  CHECK-SAME: (%[[arg0:.+]]: tensor<128xi1>, %[[arg1:.+]]: index)
//       CHECK:     %[[v0:.+]] = bufferization.alloc_tensor() copy(%[[arg0]]) {memory_space = #plan.memory_space<host_pinned>}
//       CHECK:     %[[extracted:.+]] = tensor.extract %[[v0]][%[[arg1]]] : tensor<128xi1>
//       CHECK:     return %[[extracted]]

