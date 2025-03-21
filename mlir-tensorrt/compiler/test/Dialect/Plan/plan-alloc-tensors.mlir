// RUN: mlir-tensorrt-opt %s -split-input-file -plan-alloc-tensors | FileCheck %s
// RUN: mlir-tensorrt-opt %s -split-input-file -plan-alloc-tensors=force-entrypoints-return-allocs=true  | FileCheck %s --check-prefix=CHECK-ALLOC

func.func @tensor_empty() -> tensor<128xf32> {
  %0 = tensor.empty() : tensor<128xf32>
  return %0 : tensor<128xf32>
}

// CHECK-LABEL: @tensor_empty
//  CHECK-SAME: (%[[arg0:.+]]: tensor<128xf32> {plan.result_arg}) -> tensor<128xf32> {
//       CHECK:     return %[[arg0]] : tensor<128xf32>

// CHECK-ALLOC-LABEL: @tensor_empty
//  CHECK-ALLOC-SAME: () -> tensor<128xf32> {
//       CHECK-ALLOC:     %[[v0:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<device>} : tensor<128xf32>
//       CHECK-ALLOC:     return %[[v0]] : tensor<128xf32>


// -----

func.func @from_elements(%arg0: f32, %arg1: f32) -> tensor<2xf32> {
  %0 = tensor.from_elements %arg0, %arg1 : tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func.func @from_elements
//  CHECK-SAME: (%[[arg0:.+]]: f32, %[[arg1:.+]]: f32, %[[arg2:.+]]: tensor<2xf32> {plan.result_arg}) -> tensor<2xf32> {
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[v0:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<host_pinned>} : tensor<2xf32, #plan.memory_space<host_pinned>>
//   CHECK-DAG:     %[[inserted:.+]] = tensor.insert %[[arg0]] into %[[v0]][%[[c0]]] : tensor<2xf32, #plan.memory_space<host_pinned>>
//   CHECK-DAG:     %[[inserted_0:.+]] = tensor.insert %[[arg1]] into %[[inserted]][%[[c1]]] : tensor<2xf32, #plan.memory_space<host_pinned>>
//   CHECK-DAG:     %[[cast:.+]] = tensor.cast %[[arg2]] : tensor<2xf32> to tensor<2xf32, #plan.memory_space<device>>
//       CHECK:     %[[v1:.+]] = bufferization.materialize_in_destination %[[inserted_0]] in %[[cast]] : (tensor<2xf32, #plan.memory_space<host_pinned>>, tensor<2xf32, #plan.memory_space<device>>) -> tensor<2xf32, #plan.memory_space<device>>
//       CHECK:     %[[cast_1:.+]] = tensor.cast %[[v1]] : tensor<2xf32, #plan.memory_space<device>> to tensor<2xf32>
//       CHECK:     return %[[cast_1]] : tensor<2xf32>

// CHECK-ALLOC-LABEL: func.func @from_elements
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: f32, %[[arg1:.+]]: f32) -> tensor<2xf32> {
//       CHECK-ALLOC:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK-ALLOC:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK-ALLOC:     %[[v0:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<host_pinned>} : tensor<2xf32, #plan.memory_space<host_pinned>>
//       CHECK-ALLOC:     %[[inserted:.+]] = tensor.insert %[[arg0]] into %[[v0]][%[[c0]]] : tensor<2xf32, #plan.memory_space<host_pinned>>
//       CHECK-ALLOC:     %[[inserted_0:.+]] = tensor.insert %[[arg1]] into %[[inserted]][%[[c1]]] : tensor<2xf32, #plan.memory_space<host_pinned>>
//       CHECK-ALLOC:     %[[v1:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<device>} : tensor<2xf32, #plan.memory_space<device>>
//       CHECK-ALLOC:     %[[v2:.+]] = bufferization.materialize_in_destination %[[inserted_0]] in %[[v1]] : (tensor<2xf32, #plan.memory_space<host_pinned>>, tensor<2xf32, #plan.memory_space<device>>) -> tensor<2xf32, #plan.memory_space<device>>
//       CHECK-ALLOC:     %[[cast_0:.+]] = tensor.cast %[[v2]] : tensor<2xf32, #plan.memory_space<device>> to tensor<2xf32>
//       CHECK-ALLOC:     return %[[cast_0]] : tensor<2xf32>

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

// CHECK-ALLOC-LABEL: @test_simple_dps
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<10xf32>) -> tensor<10xf32>
//   CHECK-ALLOC: %[[v0:.+]] = bufferization.alloc_tensor()
//       CHECK-ALLOC: %[[v1:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<10xf32>) outs(%[[v0]] : tensor<10xf32>)
//       CHECK-ALLOC: return %[[v1]] : tensor<10xf32>

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

// CHECK-ALLOC-LABEL: @test_two_returns
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>)
//   CHECK-ALLOC: %[[out1:.+]] = bufferization.alloc_tensor()
//   CHECK-ALLOC: %[[out2:.+]] = bufferization.alloc_tensor()
//       CHECK-ALLOC: %[[v1:.+]]:2 = linalg.generic {{.*}} ins(%[[arg0]], %[[arg1]] : {{.*}}) outs(%[[out2]], %[[out1]] : {{.*}})
//       CHECK-ALLOC: return %[[v1]]#0, %[[v1]]#1 : tensor<10xf32>, tensor<10xf32>

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
//  CHECK-DAG: %[[out0:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<device>}
//  CHECK-DAG: %[[out1:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<device>}
//  CHECK-NEXT: %[[v1:.+]]:2 = {{.*}} ins(%[[arg0]], %[[arg1]] : {{.*}}) outs(%[[out1]], %[[out0]] : {{.*}})
//       CHECK: %[[v2:.+]] = arith.addf %[[v1]]#0, %[[v1]]#1 : tensor<10xf32>
//  CHECK-NEXT: %[[v3:.+]] = bufferization.materialize_in_destination %[[v2]] in %[[arg2]] :
//  CHECK-NEXT: return %[[v3]] : tensor<10xf32>

// CHECK-ALLOC-LABEL: @main
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32>) -> tensor<10xf32>
//  CHECK-ALLOC-NEXT: %[[out0:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<device>} : tensor<10xf32>
//  CHECK-ALLOC-NEXT: %[[out1:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<device>} : tensor<10xf32>
//  CHECK-ALLOC-NEXT: %[[v1:.+]]:2 = {{.*}} ins(%[[arg0]], %[[arg1]] : {{.*}}) outs(%[[out1]], %[[out0]] : {{.*}})
//       CHECK-ALLOC: %[[v2:.+]] = arith.addf %[[v1]]#0, %[[v1]]#1 : tensor<10xf32>
//  CHECK-ALLOC-NEXT: return %[[v2]] : tensor<10xf32>


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

//  CHECK-ALLOC-LABEL: @main
//   CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>)
//        CHECK-ALLOC: %[[v0:.+]] = bufferization.alloc_tensor()
//        CHECK-ALLOC: %[[v1:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<10xf32>) outs(%[[v0]] : tensor<10xf32>)
//        CHECK-ALLOC: %[[v2:.+]] = arith.addf %[[v1]], %[[arg0]] : tensor<10xf32>
//   CHECK-ALLOC-NEXT: return %[[v2]], %[[v1]] : tensor<10xf32>, tensor<10xf32>

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

// CHECK-ALLOC-LABEL: @test_dps_chain
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<10xf32>) -> tensor<10xf32>
//   CHECK-ALLOC: %[[v0:.+]] = bufferization.alloc_tensor()
//       CHECK-ALLOC: %[[v1:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<10xf32>) outs(%[[v0]] : tensor<10xf32>)
//       CHECK-ALLOC: %[[v2:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<10xf32>) outs(%[[v1]] : tensor<10xf32>)
//       CHECK-ALLOC: return %[[v2]] : tensor<10xf32>

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

// CHECK-ALLOC-LABEL: @test_repeat_returns
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32>)
//       CHECK-ALLOC: %[[out0:.+]] = bufferization.alloc_tensor()
//       CHECK-ALLOC: %[[out1:.+]] = bufferization.alloc_tensor()
//       CHECK-ALLOC: %[[v1:.+]]:2 = linalg.generic {{.*}} ins(%[[arg0]], %[[arg1]] : {{.*}}) outs(%[[out1]], %[[out0]] : {{.*}})
//       CHECK-ALLOC: %[[v2:.+]] = arith.addf %[[v1]]#0, %[[arg0]] : tensor<10xf32>
//  CHECK-ALLOC-NEXT: return %[[v1]]#0, %[[v1]]#1, %[[v1]]#0, %[[v1]]#1, %[[v2]] : tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>



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

// CHECK-ALLOC-LABEL: @test_dps_chain_repeat
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>)
//       CHECK-ALLOC: %[[v0:.+]] = bufferization.alloc_tensor()
//       CHECK-ALLOC: %[[v1:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<10xf32>) outs(%[[v0]] : tensor<10xf32>)
//       CHECK-ALLOC: %[[v2:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<10xf32>) outs(%[[v1]] : tensor<10xf32>)
//       CHECK-ALLOC: return %[[v2]], %[[v1]], %[[v1]] : tensor<10xf32>, tensor<10xf32>, tensor<10xf32>


// -----

func.func @test_return_arg(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  return %arg0: tensor<10xf32>
}

// CHECK-LABEL: @test_return_arg
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32> {plan.result_arg})
//       CHECK:     %[[v0:.+]] = bufferization.materialize_in_destination %[[arg0]] in %[[arg1]]
//       CHECK:     return %[[v0]]

// CHECK-ALLOC-LABEL: @test_return_arg
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<10xf32>)
//  CHECK-ALLOC-NEXT:     return %[[arg0]]



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

// CHECK-ALLOC-LABEL: @test_already_dps
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32> {plan.result_arg}, %[[arg2:.+]]: tensor<10xf32> {plan.result_arg})
//       CHECK-ALLOC:     %[[v0:.+]] = bufferization.materialize_in_destination %[[arg0]] in %[[arg1]]
//       CHECK-ALLOC:     %[[v1:.+]] = bufferization.materialize_in_destination %[[arg0]] in %[[arg2]]
//       CHECK-ALLOC:     return %[[v0]], %[[v1]] : tensor<10xf32>, tensor<10xf32>

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

// CHECK-ALLOC-LABEL: @test_loop_region_dps_rewrite_while
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<10xf32>) -> tensor<10xf32> {
//   CHECK-ALLOC-DAG:     %[[c0:.+]] = arith.constant 0 :
//   CHECK-ALLOC-DAG:     %[[cst:.+]] = arith.constant 0.000000e+00
//       CHECK-ALLOC:     %[[v0:.+]] = scf.while (%[[argB:.+]] = %[[arg0]])
//       CHECK-ALLOC:       %[[v2:.+]] = bufferization.alloc_tensor() copy(%[[argB]])
//       CHECK-ALLOC:       %[[extracted:.+]] = tensor.extract %[[v2]][%[[c0]]] : tensor<10xf32>
//       CHECK-ALLOC:       %[[v3:.+]] = arith.cmpf ogt, %[[extracted]], %[[cst]] : f32
//       CHECK-ALLOC:       scf.condition(%[[v3]]) %[[argB]] : tensor<10xf32>
//       CHECK-ALLOC:     } do {
//       CHECK-ALLOC:     ^bb0(%[[arg3:.+]]: tensor<10xf32>):
//       CHECK-ALLOC:       %[[mapped:.+]] = linalg.map { math.exp } ins(%[[arg3]] : tensor<10xf32>) outs(%[[arg3]] :
//       CHECK-ALLOC:       scf.yield %[[mapped]] :
//       CHECK-ALLOC:     return %[[v0]] :

// -----

// This test is like the above, but the 'scf.while' does not forward
// all arguments from the "before" region to the "after" region. This
// is a pattern that can appear when the condition is purely calculated
// from computation in the "after" region.
// Note that bufferization will still have a hard time unless we add back the
// yielded values which were canonicalizes away (the i1 in the before region
// and results).

func.func @test_loop_region_dps_rewrite_while_arg_mismatch(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %c0 = arith.constant 0 : index
  %c0f = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<10xf32>
  %false = arith.constant 0 : i1
  %r:2 = scf.while(%first = %0, %arg1 = %false, %arg2 = %arg0)
            : (tensor<10xf32>, i1, tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
    scf.condition(%arg1) %first, %arg2 : tensor<10xf32>, tensor<10xf32>
  } do {
  ^bb0(%arg2: tensor<10xf32>, %arg4: tensor<10xf32>):
    %1 = linalg.map {math.exp}
      ins(%arg4 : tensor<10xf32>)
      outs(%arg4 : tensor<10xf32>)
    %v0 = tensor.extract %1[%c0] : tensor<10xf32>
    %cond = arith.cmpf ogt, %v0, %c0f : f32
    scf.yield %0, %cond, %1 : tensor<10xf32>, i1, tensor<10xf32>
  }
  return %r#1 : tensor<10xf32>
}


// CHECK-LABEL: func.func @test_loop_region_dps_rewrite_while_arg_mismatch
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32> {plan.result_arg})
//   CHECK-DAG:     %[[false:.+]] = arith.constant false
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[cst:.+]] = arith.constant 0.{{0.*}} : f32
//   CHECK-DAG:     %[[v0:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<device>}
//   CHECK-DAG:     %[[v1]]:2 = scf.while (%[[arg2:.+]] = %[[v0:.+]], %[[arg3:.+]] = %[[false:.+]], %[[arg4:.+]] = %[[arg0]])
//       CHECK:       scf.condition(%[[arg3]]) %[[arg2]], %[[arg4]] : tensor<10xf32>, tensor<10xf32>
//  CHECK-NEXT:     } do {
//  CHECK-NEXT:     ^bb0(%[[arg2:.+]]: tensor<10xf32>, %[[arg3:.+]]: tensor<10xf32>):
//   CHECK-DAG:       %[[mapped:.+]] = linalg.map { math.exp } ins(%[[arg3]] : tensor<10xf32>) outs(%[[arg3]] : tensor<10xf32>)
//   CHECK-DAG:       %[[extracted:.+]] = tensor.extract %[[mapped]][%[[c0]]] : tensor<10xf32>
//   CHECK-DAG:       %[[v3:.+]] = arith.cmpf ogt, %[[extracted]], %[[cst]] : f32
//   CHECK-DAG:       scf.yield %[[arg2]], %[[v3]], %[[mapped]] : tensor<10xf32>, i1, tensor<10xf32>
//  CHECK-NEXT:     }
//   CHECK-DAG:     %[[v2:.+]] = bufferization.materialize_in_destination %[[v1]]#1 in %[[arg1]] : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
//   CHECK-DAG:     return %[[v2]] : tensor<10xf32>

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

// CHECK-ALLOC-LABEL: @test_loop_region_dps_rewrite_for
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<10xf32>)
//       CHECK-ALLOC:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK-ALLOC:     %[[c10:.+]] = arith.constant 10 : index
//       CHECK-ALLOC:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK-ALLOC:     %[[v0:.+]] = scf.for %[[arg2:.+]] = %[[c0]] to %[[c10]] step %[[c1]] iter_args(%[[arg3:.+]] = %[[arg0]])
//       CHECK-ALLOC:       %[[mapped:.+]] = linalg.map { math.exp } ins(%[[arg3]] : tensor<10xf32>) outs(%[[arg3]] : tensor<10xf32>)
//       CHECK-ALLOC:       scf.yield %[[mapped]] : tensor<10xf32>
//       CHECK-ALLOC:     return %[[v0]]


// -----

func.func @alloc_tensors_from_elements(%arg0: i32) -> (tensor<1xi32> {tensorrt.host_tensor}, tensor<1xi32>) {
  %0 = tensor.from_elements %arg0 : tensor<1xi32>
  %1 = tensor.from_elements %arg0 : tensor<1xi32>
  return %0, %1 : tensor<1xi32>, tensor<1xi32>
}

// CHECK-LABEL: func.func @alloc_tensors_from_elements
//  CHECK-SAME: (%[[arg0:.+]]: i32, %[[arg1:.+]]: tensor<1xi32> {plan.result_arg}, %[[arg2:.+]]: tensor<1xi32> {plan.result_arg}) -> (tensor<1xi32> {tensorrt.host_tensor}, tensor<1xi32>) {
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[cast:.+]] = tensor.cast %[[arg2]] : tensor<1xi32> to tensor<1xi32, #plan.memory_space<device>>
//   CHECK-DAG:     %[[cast_0:.+]] = tensor.cast %[[arg1]] : tensor<1xi32> to tensor<1xi32, #plan.memory_space<host_pinned>>
//   CHECK-DAG:     %[[inserted:.+]] = tensor.insert %[[arg0]] into %[[cast_0]][%[[c0]]] : tensor<1xi32, #plan.memory_space<host_pinned>>
//   CHECK-DAG:     %[[cast_1:.+]] = tensor.cast %[[inserted]] : tensor<1xi32, #plan.memory_space<host_pinned>> to tensor<1xi32>
//   CHECK-DAG:     %[[v0:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<host_pinned>} : tensor<1xi32, #plan.memory_space<host_pinned>>
//   CHECK-DAG:     %[[inserted_2:.+]] = tensor.insert %[[arg0]] into %[[v0]][%[[c0]]] : tensor<1xi32, #plan.memory_space<host_pinned>>
//   CHECK-DAG:     %[[v1:.+]] = bufferization.materialize_in_destination %[[inserted_2]] in %[[cast]] :
//   CHECK-DAG:     %[[cast_3:.+]] = tensor.cast %[[v1]] : tensor<1xi32, #plan.memory_space<device>> to tensor<1xi32>
//       CHECK:     return %[[cast_1]], %[[cast_3]] : tensor<1xi32>, tensor<1xi32>

// CHECK-ALLOC-LABEL: func.func @alloc_tensors_from_elements
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: i32) -> (tensor<1xi32> {tensorrt.host_tensor}, tensor<1xi32>) {
//       CHECK-ALLOC:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK-ALLOC:     %[[v0:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<host_pinned>}
//       CHECK-ALLOC:     %[[inserted:.+]] = tensor.insert %[[arg0]] into %[[v0]][%[[c0]]] :
//       CHECK-ALLOC:     %[[cast:.+]] = tensor.cast %[[inserted]] : tensor<1xi32, #plan.memory_space<host_pinned>> to tensor<1xi32>
//       CHECK-ALLOC:     %[[v1:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<host_pinned>}
//       CHECK-ALLOC:     %[[inserted_0:.+]] = tensor.insert %[[arg0]] into %[[v1]][%[[c0]]]
//       CHECK-ALLOC:     %[[v2:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<device>}
//       CHECK-ALLOC:     %[[v3:.+]] = bufferization.materialize_in_destination %[[inserted_0]] in %[[v2]] :
//       CHECK-ALLOC:     %[[cast_1:.+]] = tensor.cast %[[v3]] : tensor<1xi32, #plan.memory_space<device>> to tensor<1xi32>
//       CHECK-ALLOC:     return %[[cast]], %[[cast_1]] : tensor<1xi32>, tensor<1xi32>



// -----

func.func @alloc_tensors_from_elements(%arg0: i32) -> (tensor<1xi32> {tensorrt.host_tensor}, tensor<1xi32>) {
  %0 = tensor.from_elements %arg0 : tensor<1xi32>
  return %0, %0 : tensor<1xi32>, tensor<1xi32>
}

// CHECK-LABEL: func.func @alloc_tensors_from_elements
//  CHECK-SAME: (%[[arg0:.+]]: i32, %[[arg1:.+]]: tensor<1xi32> {plan.result_arg}, %[[arg2:.+]]: tensor<1xi32> {plan.result_arg}) -> (tensor<1xi32> {tensorrt.host_tensor}, tensor<1xi32>) {
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[cast:.+]] = tensor.cast %[[arg2]] : tensor<1xi32> to tensor<1xi32, #plan.memory_space<device>>
//       CHECK:     %[[cast_0:.+]] = tensor.cast %[[arg1]] : tensor<1xi32> to tensor<1xi32, #plan.memory_space<host_pinned>>
//       CHECK:     %[[inserted:.+]] = tensor.insert %[[arg0]] into %[[cast_0]][%[[c0]]] : tensor<1xi32, #plan.memory_space<host_pinned>>
//       CHECK:     %[[cast_1:.+]] = tensor.cast %[[inserted]] : tensor<1xi32, #plan.memory_space<host_pinned>> to tensor<1xi32>
//       CHECK:     %[[v0:.+]] = bufferization.materialize_in_destination %[[inserted]] in %[[cast]] : (tensor<1xi32, #plan.memory_space<host_pinned>>, tensor<1xi32, #plan.memory_space<device>>) -> tensor<1xi32, #plan.memory_space<device>>
//       CHECK:     %[[cast_2:.+]] = tensor.cast %[[v0]] : tensor<1xi32, #plan.memory_space<device>> to tensor<1xi32>
//       CHECK:     return %[[cast_1]], %[[cast_2]] : tensor<1xi32>, tensor<1xi32>

// CHECK-ALLOC-LABEL: func.func @alloc_tensors_from_elements
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: i32) -> (tensor<1xi32> {tensorrt.host_tensor}, tensor<1xi32>) {
//       CHECK-ALLOC:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK-ALLOC:     %[[v0:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<host_pinned>} : tensor<1xi32, #plan.memory_space<host_pinned>>
//       CHECK-ALLOC:     %[[inserted:.+]] = tensor.insert %[[arg0]] into %[[v0]][%[[c0]]]
//       CHECK-ALLOC:     %[[cast:.+]] = tensor.cast %[[inserted]] : tensor<1xi32, #plan.memory_space<host_pinned>> to tensor<1xi32>
//       CHECK-ALLOC:     %[[v1:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<device>}
//       CHECK-ALLOC:     %[[v2:.+]] = bufferization.materialize_in_destination %[[inserted]] in %[[v1]]
//       CHECK-ALLOC:     %[[cast_1:.+]] = tensor.cast %[[v2]] : tensor<1xi32, #plan.memory_space<device>> to tensor<1xi32>
//       CHECK-ALLOC:     return %[[cast]], %[[cast_1]] : tensor<1xi32>, tensor<1xi32>
//       CHECK-ALLOC:   }


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
//       CHECK:     %[[v0:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<host>}
//       CHECK:     %[[inserted:.+]] = tensor.insert %[[c1]] into %[[v0]][%[[c0]]] :
//       CHECK:     %[[inserted_0:.+]] = tensor.insert %[[c2]] into %[[inserted]][%[[c1]]] :
//       CHECK:     %[[inserted_1:.+]] = tensor.insert %[[c3]] into %[[inserted_0]][%[[c2]]] :
//       CHECK:     %[[inserted_2:.+]] = tensor.insert %[[c4]] into %[[inserted_1]][%[[c3]]] :
//       CHECK:     %[[reshape:.+]] = tensor.reshape %[[arg0]](%[[inserted_2]]) :
//       CHECK:     %[[v1:.+]] = bufferization.materialize_in_destination %[[reshape]] in %[[arg1]] : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
//       CHECK:     return %[[v1]] : tensor<?x?x?x?xf32>

// CHECK-ALLOC-LABEL: @small_host_tensor_constant
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<?x?xf32>) -> tensor<?x?x?x?xf32> {
//       CHECK-ALLOC-DAG:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK-ALLOC-DAG:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK-ALLOC-DAG:     %[[c2:.+]] = arith.constant 2 : index
//       CHECK-ALLOC-DAG:     %[[c3:.+]] = arith.constant 3 : index
//       CHECK-ALLOC-DAG:     %[[c4:.+]] = arith.constant 4 : index
//       CHECK-ALLOC:     %[[v0:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<host>} :
//       CHECK-ALLOC:     %[[inserted:.+]] = tensor.insert %[[c1]] into %[[v0]][%[[c0]]] :
//       CHECK-ALLOC:     %[[inserted_0:.+]] = tensor.insert %[[c2]] into %[[inserted]][%[[c1]]] :
//       CHECK-ALLOC:     %[[inserted_1:.+]] = tensor.insert %[[c3]] into %[[inserted_0]][%[[c2]]] :
//       CHECK-ALLOC:     %[[inserted_2:.+]] = tensor.insert %[[c4]] into %[[inserted_1]][%[[c3]]] :
//       CHECK-ALLOC:     %[[reshape:.+]] = tensor.reshape %[[arg0]](%[[inserted_2]]) :
//       CHECK-ALLOC:     return %[[reshape]] : tensor<?x?x?x?xf32>


// -----

func.func @small_host_and_device_tensor_constant(%arg0: tensor<?x?xf32>) -> (tensor<?x?x?x?xf32>, tensor<4xindex>) {
  %0 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xindex>
  %1 = tensor.reshape %arg0 (%0) : (tensor<?x?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  return %1, %0 : tensor<?x?x?x?xf32>, tensor<4xindex>
}

// CHECK-LABEL: func.func @small_host_and_device_tensor_constant
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?xf32>, %[[arg1:.+]]: tensor<?x?x?x?xf32> {plan.result_arg}, %[[arg2:.+]]: tensor<4xindex> {plan.result_arg}) -> (tensor<?x?x?x?xf32>, tensor<4xindex>) {
//   CHECK-DAG:     %[[c4:.+]] = arith.constant 4 : index
//   CHECK-DAG:     %[[c3:.+]] = arith.constant 3 : index
//   CHECK-DAG:     %[[c2:.+]] = arith.constant 2 : index
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[cst:.+]] = arith.constant dense<[1, 2, 3, 4]> : tensor<4xindex>
//       CHECK:     %[[v0:.+]] = bufferization.materialize_in_destination %[[cst]] in %[[arg2]] : (tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
//       CHECK:     %[[v1:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<host>} : tensor<4xindex, #plan.memory_space<host>>
//   CHECK-DAG:     %[[inserted:.+]] = tensor.insert %[[c1]] into %[[v1]][%[[c0]]] : tensor<4xindex, #plan.memory_space<host>>
//   CHECK-DAG:     %[[inserted_0:.+]] = tensor.insert %[[c2]] into %[[inserted]][%[[c1]]] : tensor<4xindex, #plan.memory_space<host>>
//   CHECK-DAG:     %[[inserted_1:.+]] = tensor.insert %[[c3]] into %[[inserted_0]][%[[c2]]] : tensor<4xindex, #plan.memory_space<host>>
//   CHECK-DAG:     %[[inserted_2:.+]] = tensor.insert %[[c4]] into %[[inserted_1]][%[[c3]]] : tensor<4xindex, #plan.memory_space<host>>
//       CHECK:     %[[reshape:.+]] = tensor.reshape %[[arg0]](%[[inserted_2]]) : (tensor<?x?xf32>, tensor<4xindex, #plan.memory_space<host>>) -> tensor<?x?x?x?xf32>
//       CHECK:     %[[v2:.+]] = bufferization.materialize_in_destination %[[reshape]] in %[[arg1]] : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
//       CHECK:     return %[[v2]], %[[v0]] : tensor<?x?x?x?xf32>, tensor<4xindex>


// CHECK-ALLOC-LABEL: @small_host_and_device_tensor_constant
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<?x?xf32>) -> (tensor<?x?x?x?xf32>, tensor<4xindex>) {
//   CHECK-ALLOC-DAG:     %[[cst:.+]] = arith.constant dense<[1, 2, 3, 4]> : tensor<4xindex>
//   CHECK-ALLOC-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-ALLOC-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-ALLOC-DAG:     %[[c2:.+]] = arith.constant 2 : index
//   CHECK-ALLOC-DAG:     %[[c3:.+]] = arith.constant 3 : index
//   CHECK-ALLOC-DAG:     %[[c4:.+]] = arith.constant 4 : index
//   CHECK-ALLOC-DAG:     %[[v0:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<host>} :
//       CHECK-ALLOC:     %[[inserted:.+]] = tensor.insert %[[c1]] into %[[v0]][%[[c0]]] :
//       CHECK-ALLOC:     %[[inserted_0:.+]] = tensor.insert %[[c2]] into %[[inserted]][%[[c1]]] :
//       CHECK-ALLOC:     %[[inserted_1:.+]] = tensor.insert %[[c3]] into %[[inserted_0]][%[[c2]]] :
//       CHECK-ALLOC:     %[[inserted_2:.+]] = tensor.insert %[[c4]] into %[[inserted_1]][%[[c3]]] :
//       CHECK-ALLOC:     %[[reshape:.+]] = tensor.reshape %[[arg0]](%[[inserted_2]]) :
//       CHECK-ALLOC:     return %[[reshape]], %[[cst]] :

// -----

func.func @big_host_tensor_constant(%arg0: tensor<?x?xf32>) -> (tensor<1024xindex> {tensorrt.host_tensor}) {
  %0 = arith.constant dense<1> : tensor<1024xindex>
  return %0 : tensor<1024xindex>
}

// CHECK-LABEL: @big_host_tensor_constant
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?xf32>, %[[arg1:.+]]: tensor<1024xindex> {plan.result_arg}) -> (tensor<1024xindex> {tensorrt.host_tensor})
//       CHECK:     %[[cst:.+]] = arith.constant dense<1> : tensor<1024xindex>
//       CHECK:     %[[v1:.+]] = bufferization.materialize_in_destination %[[cst]] in %[[arg1]]
//       CHECK:     return %[[v1]] : tensor<1024xindex>

// CHECK-ALLOC-LABEL: @big_host_tensor_constant
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<?x?xf32>) -> (tensor<1024xindex> {tensorrt.host_tensor})
//       CHECK-ALLOC:     %[[cst:.+]] = arith.constant dense<1> : tensor<1024xindex>
//       CHECK-ALLOC:     %[[v0:.+]] = bufferization.alloc_tensor() copy(%[[cst]]) {memory_space = #plan.memory_space<host_pinned>}
//       CHECK-ALLOC:     return %[[v0]] : tensor<1024xindex>

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

// CHECK-ALLOC-LABEL: @device_extract
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<128xi1>, %[[arg1:.+]]: index)
//       CHECK-ALLOC:     %[[v0:.+]] = bufferization.alloc_tensor() copy(%[[arg0]]) {memory_space = #plan.memory_space<host_pinned>}
//       CHECK-ALLOC:     %[[extracted:.+]] = tensor.extract %[[v0]][%[[arg1]]] : tensor<128xi1>
//       CHECK-ALLOC:     return %[[extracted]]

// -----

// CHECK-LABEL: @dont_modify_nested_modules
module @dont_modify_nested_modules {

  func.func @nested() -> tensor<f32> {
    // CHECK: tensor.empty
    %0 = tensor.empty() : tensor<f32>
    return %0 : tensor<f32>
  }
}

// CHECK: @outer1
func.func @outer1() -> tensor<f32> {
  // CHECK-NOT: tensor.empty
  %0 = tensor.empty() : tensor<f32>
  return %0 : tensor<f32>
}

// CHECK: @outer2
func.func @outer2() -> tensor<f32> {
  // CHECK-NOT: tensor.empty
  %0 = tensor.empty() : tensor<f32>
  return %0 : tensor<f32>
}
