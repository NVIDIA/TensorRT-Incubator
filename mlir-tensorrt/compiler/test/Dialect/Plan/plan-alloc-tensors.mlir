// RUN: mlir-tensorrt-opt %s -split-input-file -plan-alloc-tensors=force-entrypoints-return-allocs=true  | FileCheck %s --check-prefix=CHECK-ALLOC
// RUN: mlir-tensorrt-opt %s -split-input-file -plan-alloc-tensors | FileCheck %s


func.func @tensor_empty() -> tensor<128xf32, #plan.memory_space<device>> {
  %0 = tensor.empty() : tensor<128xf32, #plan.memory_space<device>>
  return %0 : tensor<128xf32, #plan.memory_space<device>>
}

// CHECK-LABEL: @tensor_empty
//  CHECK-SAME: (%[[arg0:.+]]: tensor<{{.*}}> {plan.result_arg}) -> tensor<{{.*}}> {
//       CHECK:     return %[[arg0]] : tensor<{{.*}}>

// CHECK-ALLOC-LABEL: func.func @tensor_empty
//    CHECK-ALLOC:     %[[v0:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<device>} :
//    CHECK-ALLOC:     return %[[v0]] : tensor<128xf32, #plan.memory_space<device>>


// -----

func.func private @some_func(%arg0: tensor<128xf32, #plan.memory_space<host_pinned>>)

func.func @empty_tensor_with_encoding() {
  %0 = tensor.empty() : tensor<128xf32, #plan.memory_space<host_pinned>>
  func.call @some_func(%0) : (tensor<128xf32, #plan.memory_space<host_pinned>>) -> ()
  return
}

// CHECK-LABEL: func.func @empty_tensor_with_encoding
//       CHECK:     %[[v0:.+]] = bufferization.alloc_tensor()
// CHECK-SAME:       memory_space = #plan.memory_space<host_pinned>
// CHECK-SAME:       : tensor<128xf32, #plan.memory_space<host_pinned>>
//  CHECK-NEXT:     call @some_func(%[[v0]]) : (tensor<128xf32, #plan.memory_space<host_pinned>>)
//  CHECK-NEXT:     return

// -----

!tensor_type = tensor<10xf32, #plan.memory_space<device>>

#map = affine_map<(d0)->(d0)>
func.func @test_simple_dps(%arg0: !tensor_type) -> !tensor_type {
  %empty = tensor.empty () : !tensor_type
  %0 = linalg.generic {
    iterator_types = ["parallel"],
    indexing_maps = [#map, #map]
  } ins(%arg0: !tensor_type) outs(%empty: !tensor_type) {
    ^bb0(%a: f32, %b: f32):
      %r = arith.negf %a : f32
      linalg.yield %r : f32
  } -> !tensor_type
  return %0 : !tensor_type
}

// CHECK-LABEL: @test_simple_dps
//  CHECK-SAME: (%[[arg0:.+]]: tensor<{{.*}}>, %[[arg1:.+]]: tensor<{{.*}}> {plan.result_arg}) -> tensor<{{.*}}>
//   CHECK-NOT: bufferization.alloc_tensor()
//       CHECK: %[[v0:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<{{.*}}>) outs(%[[arg1]] : tensor<{{.*}}>)
//       CHECK: return %[[v0]] : tensor<{{.*}}>

// CHECK-ALLOC-LABEL: @test_simple_dps
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<10xf32, {{.*}}>) -> tensor<10xf32, {{.*}}>
//   CHECK-ALLOC: %[[v0:.+]] = bufferization.alloc_tensor()
//       CHECK-ALLOC: %[[v1:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<{{.*}}>) outs(%[[v0]] : tensor<{{.*}}>)
//       CHECK-ALLOC: return %[[v1]] : tensor<{{.*}}>

// -----

!tensor_type = tensor<10xf32, #plan.memory_space<device>>

#map = affine_map<(d0)->(d0)>
func.func @test_two_returns(%arg0: !tensor_type, %arg1: !tensor_type) -> (!tensor_type, !tensor_type) {
  %empty = tensor.empty () : !tensor_type
  %0, %1 = linalg.generic {
    iterator_types = ["parallel"],
    indexing_maps = [#map, #map, #map, #map]
  } ins(%arg0, %arg1: !tensor_type, !tensor_type) outs(%empty, %empty: !tensor_type, !tensor_type) {
    ^bb0(%a: f32, %b: f32, %c: f32, %d: f32):
      %r1 = arith.negf %a : f32
      %r2 = arith.negf %b : f32
      linalg.yield %r1, %r2 : f32, f32
  } -> (!tensor_type, !tensor_type)
  return %0, %1 : !tensor_type, !tensor_type
}

// CHECK-LABEL: @test_two_returns
//  CHECK-SAME: (%[[arg0:.+]]: tensor<{{.*}}>, %[[arg1:.+]]: tensor<{{.*}}>, %[[arg2:.+]]: tensor<{{.*}}> {plan.result_arg}, %[[arg3:.+]]: tensor<{{.*}}> {plan.result_arg}) -> (tensor<{{.*}}>, tensor<{{.*}}>)
//   CHECK-NOT: bufferization.alloc_tensor()
//       CHECK: %[[v0:.+]]:2 = linalg.generic {{.*}} ins(%[[arg0]], %[[arg1]] : {{.*}}) outs(%[[arg2]], %[[arg3]] : {{.*}})
//       CHECK: return %[[v0]]#0, %[[v0]]#1 : tensor<{{.*}}>, tensor<{{.*}}>

// CHECK-ALLOC-LABEL: @test_two_returns
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<{{.*}}>, %[[arg1:.+]]: tensor<{{.*}}>) -> (tensor<{{.*}}>, tensor<{{.*}}>)
//   CHECK-ALLOC: %[[out1:.+]] = bufferization.alloc_tensor()
//   CHECK-ALLOC: %[[out2:.+]] = bufferization.alloc_tensor()
//       CHECK-ALLOC: %[[v1:.+]]:2 = linalg.generic {{.*}} ins(%[[arg0]], %[[arg1]] : {{.*}}) outs(%[[out2]], %[[out1]] : {{.*}})
//       CHECK-ALLOC: return %[[v1]]#0, %[[v1]]#1 : tensor<{{.*}}>, tensor<{{.*}}>

// -----

#map = affine_map<(d0)->(d0)>
module @test_no_dps_return {
  func.func @main(%arg0: tensor<10xf32, #plan.memory_space<device>>, %arg1: tensor<10xf32, #plan.memory_space<device>>) -> tensor<10xf32, #plan.memory_space<device>> {
    %empty = tensor.empty () : tensor<10xf32, #plan.memory_space<device>>
    %0, %1 = linalg.generic {
      iterator_types = ["parallel"],
      indexing_maps = [#map, #map, #map, #map]
    } ins(%arg0, %arg1: tensor<10xf32, #plan.memory_space<device>>, tensor<10xf32, #plan.memory_space<device>>) outs(%empty, %empty: tensor<10xf32, #plan.memory_space<device>>, tensor<10xf32, #plan.memory_space<device>>) {
      ^bb0(%a: f32, %b: f32, %c: f32, %d: f32):
        %r1 = arith.negf %a : f32
        %r2 = arith.negf %b : f32
        linalg.yield %r1, %r2 : f32, f32
    } -> (tensor<10xf32, #plan.memory_space<device>>, tensor<10xf32, #plan.memory_space<device>>)
    %2 = arith.addf %0, %1 : tensor<10xf32, #plan.memory_space<device>>
    return %2 : tensor<10xf32, #plan.memory_space<device>>
  }
}

// CHECK-LABEL: @main
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32, {{.*}}>, %[[arg1:.+]]: tensor<10xf32, {{.*}}>, %[[arg2:.+]]: tensor<10xf32, {{.*}}> {plan.result_arg})
//  CHECK-DAG: %[[out0:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<device>}
//  CHECK-DAG: %[[out1:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<device>}
//  CHECK-NEXT: %[[v1:.+]]:2 = {{.*}} ins(%[[arg0]], %[[arg1]] : {{.*}}) outs(%[[out1]], %[[out0]] : {{.*}})
//       CHECK: %[[v2:.+]] = arith.addf %[[v1]]#0, %[[v1]]#1 : tensor<10xf32
//  CHECK-NEXT: %[[v3:.+]] = bufferization.materialize_in_destination %[[v2]] in %[[arg2]] :
//  CHECK-NEXT: return %[[v3]] : tensor<10xf32

// CHECK-ALLOC-LABEL: @main
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<10xf32, {{.*}}>, %[[arg1:.+]]: tensor<10xf32, {{.*}}>) -> tensor<10xf32, {{.*}}>
//  CHECK-ALLOC-NEXT: %[[out0:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<device>} :
//  CHECK-ALLOC-NEXT: %[[out1:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<device>} :
//  CHECK-ALLOC-NEXT: %[[v1:.+]]:2 = {{.*}} ins(%[[arg0]], %[[arg1]] : {{.*}}) outs(%[[out1]], %[[out0]] : {{.*}})
//       CHECK-ALLOC: %[[v2:.+]] = arith.addf %[[v1]]#0, %[[v1]]#1 : tensor<10xf32
//  CHECK-ALLOC-NEXT: return %[[v2]] : tensor<10xf32


// -----

!tensor_type = tensor<10xf32, #plan.memory_space<device>>

#map = affine_map<(d0)->(d0)>
module @test_one_dps_other_no_dps_return {
  func.func @main(%arg0: !tensor_type) -> (!tensor_type, !tensor_type) {
    %empty = tensor.empty () : !tensor_type
    %0 = linalg.generic {
      iterator_types = ["parallel"],
      indexing_maps = [#map, #map]
    } ins(%arg0: !tensor_type) outs(%empty: !tensor_type) {
      ^bb0(%a: f32, %b: f32):
        %r = arith.negf %a : f32
        linalg.yield %r : f32
    } -> !tensor_type
    %1 = arith.addf %0, %arg0 : !tensor_type
    return %1, %0 : !tensor_type, !tensor_type
  }
}

//  CHECK-LABEL: @main
//   CHECK-SAME: (%[[arg0:.+]]: tensor<{{.*}}>, %[[arg1:.+]]: tensor<{{.*}}> {plan.result_arg}, %[[arg2:.+]]: tensor<{{.*}}> {plan.result_arg}) -> (tensor<{{.*}}>, tensor<{{.*}}>)
//    CHECK-NOT: bufferization.alloc_tensor()
//        CHECK: %[[v0:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<{{.*}}>) outs(%[[arg2]] : tensor<{{.*}}>)
//        CHECK: %[[v1:.+]] = arith.addf %[[v0]], %[[arg0]] : tensor<{{.*}}>
//   CHECK-NEXT: %[[v2:.+]] = bufferization.materialize_in_destination %[[v1]] in %[[arg1]] :
//   CHECK-NEXT: return %[[v2]], %[[v0]] : tensor<{{.*}}>, tensor<{{.*}}>

//  CHECK-ALLOC-LABEL: @main
//   CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<{{.*}}>) -> (tensor<{{.*}}>, tensor<{{.*}}>)
//        CHECK-ALLOC: %[[v0:.+]] = bufferization.alloc_tensor()
//        CHECK-ALLOC: %[[v1:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<{{.*}}>) outs(%[[v0]] : tensor<{{.*}}>)
//        CHECK-ALLOC: %[[v2:.+]] = arith.addf %[[v1]], %[[arg0]] : tensor<{{.*}}>
//   CHECK-ALLOC-NEXT: return %[[v2]], %[[v1]] : tensor<{{.*}}>, tensor<{{.*}}>

// -----

!tensor_type = tensor<10xf32, #plan.memory_space<device>>

#map = affine_map<(d0)->(d0)>
func.func @test_dps_chain(%arg0: !tensor_type) -> !tensor_type {
  %empty = tensor.empty () : !tensor_type
  %0 = linalg.generic {
    iterator_types = ["parallel"],
    indexing_maps = [#map, #map]
  } ins(%arg0: !tensor_type) outs(%empty: !tensor_type) {
    ^bb0(%a: f32, %b: f32):
      %r = arith.negf %a : f32
      linalg.yield %r : f32
  } -> !tensor_type
  %2 = linalg.generic {
    iterator_types = ["parallel"],
    indexing_maps = [#map, #map]
  } ins(%arg0: !tensor_type) outs(%0: !tensor_type) {
    ^bb0(%a: f32, %b: f32):
      %r = arith.negf %a : f32
      linalg.yield %r : f32
  } -> !tensor_type
  return %2 : !tensor_type
}

// CHECK-LABEL: @test_dps_chain
//  CHECK-SAME: (%[[arg0:.+]]: tensor<{{.*}}>, %[[arg1:.+]]: tensor<{{.*}}> {plan.result_arg}) -> tensor<{{.*}}>
//   CHECK-NOT: bufferization.alloc_tensor()
//       CHECK: %[[v0:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<{{.*}}>) outs(%[[arg1]] : tensor<{{.*}}>)
//       CHECK: %[[v1:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<{{.*}}>) outs(%[[v0]] : tensor<{{.*}}>)
//       CHECK: return %[[v1]] : tensor<{{.*}}>

// CHECK-ALLOC-LABEL: @test_dps_chain
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<{{.*}}>) -> tensor<{{.*}}>
//   CHECK-ALLOC: %[[v0:.+]] = bufferization.alloc_tensor()
//       CHECK-ALLOC: %[[v1:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<{{.*}}>) outs(%[[v0]] : tensor<{{.*}}>)
//       CHECK-ALLOC: %[[v2:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<{{.*}}>) outs(%[[v1]] : tensor<{{.*}}>)
//       CHECK-ALLOC: return %[[v2]] : tensor<{{.*}}>

// -----

!tensor_type = tensor<10xf32, #plan.memory_space<device>>

#map = affine_map<(d0)->(d0)>
func.func @test_repeat_returns(%arg0: !tensor_type, %arg1: !tensor_type) -> (!tensor_type, !tensor_type, !tensor_type, !tensor_type, !tensor_type) {
  %empty = tensor.empty () : !tensor_type
  %0, %1 = linalg.generic {
    iterator_types = ["parallel"],
    indexing_maps = [#map, #map, #map, #map]
  } ins(%arg0, %arg1: !tensor_type, !tensor_type) outs(%empty, %empty: !tensor_type, !tensor_type) {
    ^bb0(%a: f32, %b: f32, %c: f32, %d: f32):
      %r1 = arith.negf %a : f32
      %r2 = arith.negf %b : f32
      linalg.yield %r1, %r2 : f32, f32
  } -> (!tensor_type, !tensor_type)
  %2 = arith.addf %0, %arg0 : !tensor_type
  return %0, %1, %0, %1, %2 : !tensor_type, !tensor_type, !tensor_type, !tensor_type, !tensor_type
}

// CHECK-LABEL: @test_repeat_returns
//  CHECK-SAME: (%[[arg0:.+]]: tensor<{{.*}}>, %[[arg1:.+]]: tensor<{{.*}}>, %[[arg2:.+]]: tensor<{{.*}}> {plan.result_arg}, %[[arg3:.+]]: tensor<{{.*}}> {plan.result_arg}, %[[arg4:.+]]: tensor<{{.*}}> {plan.result_arg}, %[[arg5:.+]]: tensor<{{.*}}> {plan.result_arg}, %[[arg6:.+]]: tensor<{{.*}}> {plan.result_arg})
//   CHECK-NOT: bufferization.alloc_tensor()
//       CHECK: %[[v0:.+]]:2 = linalg.generic {{.*}} ins(%[[arg0]], %[[arg1]] : {{.*}}) outs(%[[arg2]], %[[arg3]] : {{.*}})
//       CHECK: %[[v1:.+]] = arith.addf %[[v0]]#0, %[[arg0]] : tensor<{{.*}}>
//  CHECK-NEXT: %[[v2:.+]] = bufferization.materialize_in_destination %[[v0]]#0 in %[[arg4]] :
//  CHECK-NEXT: %[[v3:.+]] = bufferization.materialize_in_destination %[[v0]]#1 in %[[arg5]] :
//  CHECK-NEXT: %[[v4:.+]] = bufferization.materialize_in_destination %[[v1]] in %[[arg6]] :
//  CHECK-NEXT: return %[[v0]]#0, %[[v0]]#1, %[[v2]], %[[v3]], %[[v4]] :

// CHECK-ALLOC-LABEL: @test_repeat_returns
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<{{.*}}>, %[[arg1:.+]]: tensor<{{.*}}>)
//       CHECK-ALLOC: %[[out0:.+]] = bufferization.alloc_tensor()
//       CHECK-ALLOC: %[[out1:.+]] = bufferization.alloc_tensor()
//       CHECK-ALLOC: %[[v1:.+]]:2 = linalg.generic {{.*}} ins(%[[arg0]], %[[arg1]] : {{.*}}) outs(%[[out1]], %[[out0]] : {{.*}})
//       CHECK-ALLOC: %[[v2:.+]] = arith.addf %[[v1]]#0, %[[arg0]]
//  CHECK-ALLOC-NEXT: return %[[v1]]#0, %[[v1]]#1, %[[v1]]#0, %[[v1]]#1, %[[v2]] :



// -----

!tensor_type = tensor<10xf32, #plan.memory_space<device>>

#map = affine_map<(d0)->(d0)>
func.func @test_dps_chain_repeat(%arg0: !tensor_type) -> (!tensor_type, !tensor_type, !tensor_type) {
  %empty = tensor.empty () : !tensor_type
  %0 = linalg.generic {
    iterator_types = ["parallel"],
    indexing_maps = [#map, #map]
  } ins(%arg0: !tensor_type) outs(%empty: !tensor_type) {
    ^bb0(%a: f32, %b: f32):
      %r = arith.negf %a : f32
      linalg.yield %r : f32
  } -> !tensor_type
  %2 = linalg.generic {
    iterator_types = ["parallel"],
    indexing_maps = [#map, #map]
  } ins(%arg0: !tensor_type) outs(%0: !tensor_type) {
    ^bb0(%a: f32, %b: f32):
      %r = arith.negf %a : f32
      linalg.yield %r : f32
  } -> !tensor_type
  return %2, %0, %0 : !tensor_type, !tensor_type, !tensor_type
}

// CHECK-LABEL: @test_dps_chain_repeat
//  CHECK-SAME: (%[[arg0:.+]]: tensor<{{.*}}>, %[[arg1:.+]]: tensor<{{.*}}> {plan.result_arg}, %[[arg2:.+]]: tensor<{{.*}}> {plan.result_arg}, %[[arg3:.+]]: tensor<{{.*}}> {plan.result_arg}) -> (tensor<{{.*}}>, tensor<{{.*}}>, tensor<{{.*}}>)
//   CHECK-NOT: bufferization.alloc_tensor()
//       CHECK: %[[v0:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<{{.*}}>) outs(%[[arg2]] : tensor<{{.*}}>)
//       CHECK: %[[v1:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<{{.*}}>) outs(%[[v0]] : tensor<{{.*}}>)
//       CHECK: %[[v2:.+]] = bufferization.materialize_in_destination %[[v1]] in %[[arg1]] :
//  CHECK-NEXT: %[[v3:.+]] = bufferization.materialize_in_destination %[[v0]] in %[[arg3]] :
//  CHECK-NEXT: return %[[v2]], %[[v0]], %[[v3]] :

// CHECK-ALLOC-LABEL: @test_dps_chain_repeat
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<{{.*}}>) -> (tensor<{{.*}}>, tensor<{{.*}}>, tensor<{{.*}}>)
//       CHECK-ALLOC: %[[v0:.+]] = bufferization.alloc_tensor()
//       CHECK-ALLOC: %[[v1:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<{{.*}}>) outs(%[[v0]] : tensor<{{.*}}>)
//       CHECK-ALLOC: %[[v2:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<{{.*}}>) outs(%[[v1]] : tensor<{{.*}}>)
//       CHECK-ALLOC: return %[[v2]], %[[v1]], %[[v1]] :


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
//  CHECK-ALLOC-NEXT:   %[[copy:.+]] = bufferization.alloc_tensor() copy(%[[arg0]])
//  CHECK-ALLOC-NEXT:   return %[[copy]]



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
//   CHECK-ALLOC-DAG:     %[[copy0:.+]] = bufferization.alloc_tensor() copy(%[[v0]])
//   CHECK-ALLOC-DAG:     %[[copy1:.+]] = bufferization.alloc_tensor() copy(%[[v1]])
//       CHECK-ALLOC:     return %[[copy0]], %[[copy1]] : tensor<10xf32>, tensor<10xf32>

// -----

// This test is like the above, but the 'scf.while' does not forward
// all arguments from the "before" region to the "after" region. This
// is a pattern that can appear when the condition is purely calculated
// from computation in the "after" region.
// Note that bufferization will still have a hard time unless we add back the
// yielded values which were canonicalizes away (the i1 in the before region
// and results).

func.func @test_loop_region_dps_rewrite_while_arg_mismatch(%arg0: tensor<10xf32, #plan.memory_space<device>>)
      -> tensor<10xf32, #plan.memory_space<device>> {
  %false = arith.constant false
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<10xf32, #plan.memory_space<device>>
  %1:2 = scf.while (%arg1 = %0, %arg2 = %false, %arg3 = %arg0)
      : (tensor<10xf32, #plan.memory_space<device>>, i1, tensor<10xf32, #plan.memory_space<device>>)
         -> (tensor<10xf32, #plan.memory_space<device>>, tensor<10xf32, #plan.memory_space<device>>) {
    scf.condition(%arg2) %arg1, %arg3 : tensor<10xf32, #plan.memory_space<device>>, tensor<10xf32, #plan.memory_space<device>>
  } do {
  ^bb0(%arg1: tensor<10xf32, #plan.memory_space<device>>, %arg2: tensor<10xf32, #plan.memory_space<device>>):
    %mapped = linalg.map { math.exp } ins(%arg2 : tensor<10xf32, #plan.memory_space<device>>)
       outs(%arg2 : tensor<10xf32, #plan.memory_space<device>>)
    %cast = tensor.cast %mapped : tensor<10xf32, #plan.memory_space<device>>
       to tensor<10xf32, #plan.memory_space<host>>
    %extracted = tensor.extract %cast[%c0] : tensor<10xf32, #plan.memory_space<host>>
    %2 = arith.cmpf ogt, %extracted, %cst : f32
    scf.yield %0, %2, %mapped : tensor<10xf32, #plan.memory_space<device>>, i1, tensor<10xf32, #plan.memory_space<device>>
  }
  return %1#1 : tensor<10xf32, #plan.memory_space<device>>
}


// CHECK-LABEL: func.func @test_loop_region_dps_rewrite_while_arg_mismatch
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32, {{.*}}>, %[[arg1:.+]]: tensor<10xf32, {{.*}}> {plan.result_arg})
//   CHECK-DAG:     %[[false:.+]] = arith.constant false
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[cst:.+]] = arith.constant 0.{{0.*}} : f32
//   CHECK-DAG:     %[[v0:.+]] = bufferization.alloc_tensor() {memory_space = #plan.memory_space<device>}
//   CHECK-DAG:     %[[v1]]:2 = scf.while (%[[arg2:.+]] = %[[v0:.+]], %[[arg3:.+]] = %[[false:.+]], %[[arg4:.+]] = %[[arg0]])
//       CHECK:       scf.condition(%[[arg3]]) %[[arg2]], %[[arg4]] :
//  CHECK-NEXT:     } do {
//  CHECK-NEXT:     ^bb0(%[[arg2:.+]]: tensor<10xf32, {{.*}}>, %[[arg3:.+]]: tensor<10xf32, {{.*}}>):
//   CHECK-DAG:       %[[mapped:.+]] = linalg.map { math.exp } ins(%[[arg3]] : tensor<10xf32, {{.*}}>) outs(%[[arg3]] :
//   CHECK-DAG:       %[[cast:.+]] = tensor.cast %[[mapped]] :
//   CHECK-DAG:       %[[extracted:.+]] = tensor.extract %[[cast]][%[[c0]]]
//   CHECK-DAG:       %[[v3:.+]] = arith.cmpf ogt, %[[extracted]], %[[cst]] : f32
//   CHECK-DAG:       scf.yield %[[arg2]], %[[v3]], %[[mapped]] :
//  CHECK-NEXT:     }
//   CHECK-DAG:     %[[v2:.+]] = bufferization.materialize_in_destination %[[v1]]#1 in %[[arg1]]
//   CHECK-DAG:     return %[[v2]] :

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
//   CHECK-ALLOC-DAG:     %[[copy:.+]] = bufferization.alloc_tensor() copy(%[[v0]])
//       CHECK-ALLOC:     return %[[copy]] :

// -----

func.func @small_host_tensor_constant(%arg0: tensor<?x?xf32, #plan.memory_space<device>>)
      -> tensor<?x?x?x?xf32, #plan.memory_space<device>> {
  %cst = arith.constant dense<[1, 2, 3, 4]> : tensor<4xindex, #plan.memory_space<host>>
  %reshape = tensor.reshape %arg0(%cst) :
     (tensor<?x?xf32, #plan.memory_space<device>>, tensor<4xindex, #plan.memory_space<host>>)
     -> tensor<?x?x?x?xf32, #plan.memory_space<device>>
  return %reshape : tensor<?x?x?x?xf32, #plan.memory_space<device>>
}

// CHECK-LABEL: func.func @small_host_tensor_constant
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?xf32, #plan.memory_space<device>>,
//  CHECK-SAME:  %[[arg1:.+]]: tensor<?x?x?x?xf32, #plan.memory_space<device>> {plan.result_arg})
//       CHECK:     %[[cst:.+]] = arith.constant dense<[1, 2, 3, 4]> : tensor<4xindex, #plan.memory_space<host>>
//       CHECK:     %[[reshape:.+]] = tensor.reshape %[[arg0]](%[[cst]]) :
//       CHECK:     %[[v0:.+]] = bufferization.materialize_in_destination %[[reshape]] in %[[arg1]] :
//       CHECK:     return %[[v0]] : tensor<?x?x?x?xf32, #plan.memory_space<device>>

// CHECK-ALLOC-LABEL: func.func @small_host_tensor_constant
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<?x?xf32, #plan.memory_space<device>>)
//       CHECK-ALLOC:     %[[cst:.+]] = arith.constant dense<[1, 2, 3, 4]> : tensor<4xindex, #plan.memory_space<host>>
//       CHECK-ALLOC:     %[[reshape:.+]] = tensor.reshape %[[arg0]](%[[cst]]) : (tensor<?x?xf32, #plan.memory_space<device>>, tensor<4xindex, #plan.memory_space<host>>) -> tensor<?x?x?x?xf32, #plan.memory_space<device>>
//       CHECK-ALLOC:     %[[v0:.+]] = bufferization.alloc_tensor() copy(%[[reshape]]) : tensor<?x?x?x?xf32, #plan.memory_space<device>>
//       CHECK-ALLOC:     return %[[v0]] : tensor<?x?x?x?xf32, #plan.memory_space<device>>

// -----

func.func @small_host_and_device_tensor_constant(%arg0: tensor<?x?xf32, #plan.memory_space<device>>)
    -> (tensor<?x?x?x?xf32, #plan.memory_space<device>>, tensor<4xindex, #plan.memory_space<device>>) {
  %cst = arith.constant dense<[1, 2, 3, 4]> : tensor<4xindex, #plan.memory_space<device>>
  %cst_0 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xindex, #plan.memory_space<host>>
  %reshape = tensor.reshape %arg0(%cst_0) :
     (tensor<?x?xf32, #plan.memory_space<device>>, tensor<4xindex, #plan.memory_space<host>>)
      -> tensor<?x?x?x?xf32, #plan.memory_space<device>>
  return %reshape, %cst : tensor<?x?x?x?xf32, #plan.memory_space<device>>, tensor<4xindex, #plan.memory_space<device>>
}

// CHECK-LABEL: func.func @small_host_and_device_tensor_constant
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?xf32, #plan.memory_space<device>>,
//  CHECK-SAME:  %[[arg1:.+]]: tensor<?x?x?x?xf32, #plan.memory_space<device>> {plan.result_arg},
//  CHECK-SAME:  %[[arg2:.+]]: tensor<4xindex, #plan.memory_space<device>> {plan.result_arg})
//   CHECK-DAG:     %[[cst:.+]] = arith.constant dense<[1, 2, 3, 4]> : tensor<4xindex, #plan.memory_space<host>>
//   CHECK-DAG:     %[[cst_0:.+]] = arith.constant dense<[1, 2, 3, 4]> : tensor<4xindex, #plan.memory_space<device>>
//   CHECK-DAG:     %[[v0:.+]] = bufferization.materialize_in_destination %[[cst_0]] in %[[arg2]] :
//   CHECK-DAG:     %[[reshape:.+]] = tensor.reshape %[[arg0]](%[[cst]])
//   CHECK-DAG:     %[[v1:.+]] = bufferization.materialize_in_destination %[[reshape]] in %[[arg1]] :
//   CHECK-DAG:     return %[[v1]], %[[v0]]

// CHECK-ALLOC-LABEL: func.func @small_host_and_device_tensor_constant
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<?x?xf32, #plan.memory_space<device>>)
// CHECK-ALLOC-DAG:     %[[cst:.+]] = arith.constant dense<[1, 2, 3, 4]> : tensor<4xindex, #plan.memory_space<device>>
// CHECK-ALLOC-DAG:     %[[cst_0:.+]] = arith.constant dense<[1, 2, 3, 4]> : tensor<4xindex, #plan.memory_space<host>>
// CHECK-ALLOC-DAG:     %[[reshape:.+]] = tensor.reshape %[[arg0]](%[[cst_0]]) : (tensor<?x?xf32, #plan.memory_space<device>>, tensor<4xindex, #plan.memory_space<host>>) -> tensor<?x?x?x?xf32, #plan.memory_space<device>>
// CHECK-ALLOC-DAG:     %[[v0:.+]] = bufferization.alloc_tensor() copy(%[[reshape]]) : tensor<?x?x?x?xf32, #plan.memory_space<device>>
// CHECK-ALLOC-DAG:     return %[[v0]], %[[cst]] :

// -----

func.func @big_host_tensor_constant() -> (tensor<1024xindex, #plan.memory_space<host>>) {
  %0 = arith.constant dense<1> : tensor<1024xindex, #plan.memory_space<host>>
  return %0 : tensor<1024xindex, #plan.memory_space<host>>
}

// CHECK-LABEL: func.func @big_host_tensor_constant
//  CHECK-SAME: (%[[arg1:.+]]: tensor<1024xindex, #plan.memory_space<host>>
//    CHECK-DAG:     %[[cst:.+]] = arith.constant dense<1> : tensor<1024xindex, #plan.memory_space<host>>
//    CHECK-DAG:     %[[v0:.+]] = bufferization.materialize_in_destination %[[cst]] in %[[arg1]]
//    CHECK-DAG:     return %[[v0]]


// CHECK-ALLOC-LABEL: func.func @big_host_tensor_constant
//  CHECK-ALLOC-SAME: () -> tensor<1024xindex, #plan.memory_space<host>>
//       CHECK-ALLOC:     %[[cst:.+]] = arith.constant dense<1> : tensor<1024xindex, #plan.memory_space<host>>
//       CHECK-ALLOC:     return %[[cst]] : tensor<1024xindex, #plan.memory_space<host>>

// -----

func.func private @ext_user(%arg0: tensor<1024xindex, #plan.memory_space<host>>)

func.func @constant_with_memory_space_encoding() {
  %cst = arith.constant dense<1> : tensor<1024xindex, #plan.memory_space<host>>
  call @ext_user(%cst) : (tensor<1024xindex, #plan.memory_space<host>>) -> ()
  return
}

// CHECK-LABEL: func.func @constant_with_memory_space_encoding
//   CHECK-DAG:   %[[cst:.+]] = arith.constant dense<1> : tensor<1024xindex, #plan.memory_space<host>>
//   CHECK-DAG:   call @ext_user(%[[cst]]) : (tensor<1024xindex, #plan.memory_space<host>>)
//   CHECK-DAG:   return

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

// -----

!type = tensor<1xi32, #plan.memory_space<device>>
!returned_type = tensor<1x1xi32, #plan.memory_space<device>>

func.func @test_dps_reshape_expand_equivalent(
    %arg0: !type, %arg1: !type) -> !returned_type {
  %0 = tensor.empty() : !type
  %1 = linalg.map {arith.addi} ins(%arg0, %arg1 : !type, !type) outs(%0 : !type)
  %2 = tensor.expand_shape %1 [[0, 1]] output_shape [1, 1]: !type into !returned_type
  return %2 : !returned_type
}

// CHECK-LABEL: func.func @test_dps_reshape_expand_equivalent
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xi32, #plan.memory_space<device>>, %[[arg1:.+]]: tensor<1xi32, #plan.memory_space<device>>, %[[arg2:.+]]: tensor<1x1xi32, #plan.memory_space<device>> {plan.result_arg})
//       CHECK:     %[[collapsed:.+]] = tensor.collapse_shape %[[arg2]] {{\[}}[0, 1]]
//       CHECK:     %[[mapped:.+]] = linalg.map {{.*}} ins(%[[arg0]], %[[arg1]] : {{.*}}) outs(%[[collapsed]] : {{.*}})
//       CHECK:     %[[expanded:.+]] = tensor.expand_shape %[[mapped]]
//       CHECK:     return %[[expanded]] : tensor<1x1xi32, #plan.memory_space<device>>

// -----

!type = tensor<?xi32, #plan.memory_space<device>>
!returned_type = tensor<?x?xi32, #plan.memory_space<device>>

func.func @test_dps_dynamic_reshape_expand_equivalent(
    %arg0: !type, %arg1: !type, %arg2: index, %arg3: index) -> !returned_type {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : !type
  %0 = tensor.empty(%dim) : !type
  %1 = linalg.map {arith.addi} ins(%arg0, %arg1 : !type, !type) outs(%0 : !type)
  %2 = tensor.expand_shape %1 [[0, 1]] output_shape [%arg2, %arg3]: !type into !returned_type
  return %2 : !returned_type
}

// CHECK-LABEL: func.func @test_dps_dynamic_reshape_expand_equivalent
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xi32, #plan.memory_space<device>>, %[[arg1:.+]]: tensor<?xi32, #plan.memory_space<device>>, %[[arg2:.+]]: index, %[[arg3:.+]]: index, %[[arg4:.+]]: tensor<?x?xi32, #plan.memory_space<device>> {plan.result_arg})
//       CHECK:     %[[collapsed:.+]] = tensor.collapse_shape %[[arg4]] {{\[}}[0, 1]] :
//       CHECK:     %[[mapped:.+]] = linalg.map {{.*}} ins(%[[arg0]], %[[arg1]] : {{.*}}) outs(%[[collapsed]] : {{.*}})
//       CHECK:     %[[expanded:.+]] = tensor.expand_shape %[[mapped]]
//       CHECK:     return %[[expanded]] : tensor<?x?xi32, #plan.memory_space<device>>

// -----

!type = tensor<2x3xi32, #plan.memory_space<device>>
!returned_type = tensor<6xi32, #plan.memory_space<device>>

func.func @test_dps_reshape_collapse_equivalent(
    %arg0: !type, %arg1: !type) -> !returned_type {
  %0 = tensor.empty() : !type
  %1 = linalg.map {arith.addi} ins(%arg0, %arg1 : !type, !type) outs(%0 : !type)
  %2 = tensor.collapse_shape %1 [[0, 1]] : !type into !returned_type
  return %2 : !returned_type
}

// CHECK-LABEL: func.func @test_dps_reshape_collapse_equivalent
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x3xi32{{.*}}>, %[[arg1:.+]]: tensor<2x3xi32{{.*}}>, %[[arg2:.+]]: tensor<6xi32{{.*}}>
//   CHECK-DAG:     %[[expanded:.+]] = tensor.expand_shape %[[arg2]] {{\[}}[0, 1]] output_shape [2, 3] :
//   CHECK-DAG:     %[[mapped:.+]] = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} } ins(%[[arg0]], %[[arg1]] : {{.*}}, {{.*}}) outs(%[[expanded]] : {{.*}})
//   CHECK-DAG:     %[[collapsed:.+]] = tensor.collapse_shape %[[mapped]] {{\[}}[0, 1]]
//   CHECK-DAG:     return %[[collapsed]]

// -----

!type = tensor<2x3xcomplex<f32>, #plan.memory_space<device>>
!returned_type = tensor<6xcomplex<f32>, #plan.memory_space<device>>

func.func @test_dps_complex_reshape_collapse_equivalent(
    %arg0: !type, %arg1: !type) -> !returned_type {
  %0 = tensor.empty() : !type
  %1 = linalg.map {complex.add} ins(%arg0, %arg1 : !type, !type) outs(%0 : !type)
  %2 = tensor.collapse_shape %1 [[0, 1]] : !type into !returned_type
  return %2 : !returned_type
}

// CHECK-LABEL: func.func @test_dps_complex_reshape_collapse_equivalent
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x3xcomplex<f32>, #plan.memory_space<device>>, %[[arg1:.+]]: tensor<2x3xcomplex<f32>, #plan.memory_space<device>>, %[[arg2:.+]]: tensor<6xcomplex<f32>, #plan.memory_space<device>> {plan.result_arg})
//   CHECK-DAG:     %[[expanded:.+]] = tensor.expand_shape %[[arg2]] {{\[}}[0, 1]] output_shape [2, 3] :
//   CHECK-DAG:     %[[mapped:.+]] = linalg.map { complex.add } ins(%[[arg0]], %[[arg1]] :
//   CHECK-DAG:     %[[collapsed:.+]] = tensor.collapse_shape %[[mapped]]
//   CHECK-DAG:     return %[[collapsed]]

// -----

!type = tensor<?x?xi32, #plan.memory_space<device>>
!returned_type = tensor<?xi32, #plan.memory_space<device>>

func.func @test_dps_dynamic_reshape_collapse_equivalent(
    %arg0: !type, %arg1: !type) -> !returned_type {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %arg0, %c0 : !type
  %dim1 = tensor.dim %arg0, %c1 : !type
  %0 = tensor.empty(%dim0, %dim1) : !type
  %1 = linalg.map {arith.addi} ins(%arg0, %arg1 : !type, !type) outs(%0 : !type)
  %2 = tensor.collapse_shape %1 [[0, 1]] : !type into !returned_type
  return %2 : !returned_type
}

// CHECK-LABEL: func.func @test_dps_dynamic_reshape_collapse_equivalent
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?xi32, {{.*}}>, %[[arg1:.+]]: tensor<?x?xi32, {{.*}}>, %[[arg2:.+]]: tensor<?xi32
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]] :
//   CHECK-DAG:     %[[dim_0:.+]] = tensor.dim %[[arg0]], %[[c1]] :
//   CHECK-DAG:     %[[expanded:.+]] = tensor.expand_shape %[[arg2]] {{\[}}[0, 1]] output_shape [%[[dim]], %[[dim_0]]] :
//   CHECK-DAG:     %[[mapped:.+]] = linalg.map {{.*}} ins(%[[arg0]], %[[arg1]] : {{.*}}) outs(%[[expanded]] :
//   CHECK-DAG:     %[[collapsed:.+]] = tensor.collapse_shape %[[mapped]] {{\[}}[0, 1]]
//   CHECK-DAG:     return %[[collapsed]] : tensor<?xi32

// -----

!type = tensor<?x?xi32, #plan.memory_space<device>>
!returned_type = tensor<?x1x1x?xi32, #plan.memory_space<device>>

func.func @test_dps_dynamic_reshape_ambiguous(
    %arg0: !type, %arg1: !type) -> !returned_type {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %arg0, %c0 : !type
  %dim1 = tensor.dim %arg0, %c1 : !type
  %0 = tensor.empty(%dim0, %dim1) : !type
  %1 = linalg.map {arith.addi} ins(%arg0, %arg1 : !type, !type) outs(%0 : !type)
  %3 = tensor.expand_shape %1 [[0, 1], [2, 3]]
    output_shape [%dim0, 1, 1, %dim0] : !type into !returned_type
  return %3 : !returned_type
}

// CHECK-LABEL: func.func @test_dps_dynamic_reshape_ambiguous
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?xi32, #plan.memory_space<device>>, %[[arg1:.+]]: tensor<?x?xi32, #plan.memory_space<device>>, %[[arg2:.+]]: tensor<?x1x1x?xi32, #plan.memory_space<device>> {plan.result_arg})
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]]
//   CHECK-DAG:     %[[dim_0:.+]] = tensor.dim %[[arg0]], %[[c1]]
//   CHECK-DAG:     %[[from_elements:.+]] = tensor.from_elements %[[dim]], %[[dim_0]] : tensor<2xindex, #plan.memory_space<host>>
//   CHECK-DAG:     %[[reshape:.+]] = tensor.reshape %[[arg2]](%[[from_elements]]) :
//   CHECK-DAG:     %[[mapped:.+]] = linalg.map {{.*}} ins(%[[arg0]], %[[arg1]] : {{.*}}) outs(%[[reshape]] :
//   CHECK-DAG:     %[[expanded:.+]] = tensor.expand_shape %[[mapped]]
//   CHECK-DAG:     return %[[expanded]]

// -----

!type = tensor<2xi32, #plan.memory_space<device>>
!returned_type = tensor<2xf32, #plan.memory_space<device>>

// TODO: Fix this upstream. There is not yet a way to bufferize
// `tensor.bitcast`.

func.func @test_dps_bitcast_not_equivalent(
    %arg0: !type, %arg1: !type) -> !returned_type {
  %0 = tensor.empty() : !type
  %1 = linalg.map {arith.addi} ins(%arg0, %arg1 : !type, !type) outs(%0 : !type)
  %2 = tensor.bitcast %1 : !type to !returned_type
  return %2 : !returned_type
}

// CHECK-LABEL: func.func @test_dps_bitcast_not_equivalent
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2xi32, #plan.memory_space<device>>, %[[arg1:.+]]: tensor<2xi32, #plan.memory_space<device>>, %[[arg2:.+]]: tensor<2xf32, #plan.memory_space<device>> {plan.result_arg})
//   CHECK-DAG:     %[[v0:.+]] = bufferization.alloc_tensor()
//   CHECK-DAG:     %[[mapped:.+]] = linalg.map {{.*}} ins(%[[arg0]], %[[arg1]] : {{.*}}) outs(%[[v0]] : {{.*}})
//   CHECK-DAG:     %[[v1:.+]] = tensor.bitcast %[[mapped]]
//   CHECK-DAG:     %[[v2:.+]] = bufferization.materialize_in_destination %[[v1]] in %[[arg2]]
//   CHECK-DAG:     return %[[v2]]
