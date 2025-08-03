// RUN: mlir-tensorrt-opt %s -split-input-file -plan-alloc-tensors=force-entrypoints-return-allocs=true  | FileCheck %s --check-prefix=CHECK-ALLOC
// RUN: mlir-tensorrt-opt %s -split-input-file -plan-alloc-tensors | FileCheck %s

!tensor_type = tensor<10xf32, #plan.memory_space<device>>

#map = affine_map<(d0)->(d0)>
func.func @test_simple_donation(%arg0: !tensor_type {plan.aliasing_output = 0 : i32}) -> !tensor_type {
  %empty = tensor.empty () : !tensor_type
  %0 = linalg.map {arith.negf} ins(%arg0: !tensor_type) outs(%empty: !tensor_type)
  return %0 : !tensor_type
}

// CHECK-LABEL: @test_simple_donation
//  CHECK-SAME: (%[[arg0:.+]]: tensor<{{.*}}> {plan.aliasing_output = 0 : i32}) -> tensor<{{.*}}>
//   CHECK-NOT: bufferization.alloc_tensor()
//       CHECK: %[[v0:.+]] = linalg.map { arith.negf } ins(%[[arg0]] : tensor<{{.*}}>) outs(%[[arg0]] : tensor<{{.*}}>)
//       CHECK: return %[[v0]] : tensor<{{.*}}>

// CHECK-ALLOC-LABEL: @test_simple_donation
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<10xf32, {{.*}}> {plan.aliasing_output = 0 : i32}) -> tensor<10xf32, {{.*}}>
//   CHECK-ALLOC: %[[v0:.+]] = tensor.empty() : tensor<10xf32, #plan.memory_space<device>>
//       CHECK-ALLOC: %[[v1:.+]] = linalg.map { arith.negf } ins(%[[arg0]] : tensor<{{.*}}>) outs(%[[v0]] : tensor<{{.*}}>)
//       CHECK-ALLOC: return %[[v1]] : tensor<{{.*}}>

// -----

!tensor_type = tensor<10xf32, #plan.memory_space<device>>

#map = affine_map<(d0)->(d0)>
func.func @test_one_donation_one_return(%arg0: !tensor_type {plan.aliasing_output = 1 : i32}, %arg1: !tensor_type) -> (!tensor_type, !tensor_type) {
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

// CHECK-LABEL: @test_one_donation_one_return
//  CHECK-SAME: (%[[arg0:.+]]: tensor<{{.*}}> {plan.aliasing_output = 1 : i32}, %[[arg1:.+]]: tensor<{{.*}}>, %[[arg2:.+]]: tensor<{{.*}}> {plan.result_arg}) -> (tensor<{{.*}}>, tensor<{{.*}}>)
//   CHECK-NOT: bufferization.alloc_tensor()
//       CHECK: %[[v0:.+]]:2 = linalg.generic {{.*}} ins(%[[arg0]], %[[arg1]] : {{.*}}) outs(%[[arg2]], %[[arg0]] : {{.*}})
//       CHECK: return %[[v0]]#0, %[[v0]]#1 : tensor<{{.*}}>, tensor<{{.*}}>

// CHECK-ALLOC-LABEL: @test_one_donation_one_return
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<{{.*}}> {plan.aliasing_output = 1 : i32}, %[[arg1:.+]]: tensor<{{.*}}>) -> (tensor<{{.*}}>, tensor<{{.*}}>)
//   CHECK-ALLOC-DAG: %[[out1:.+]] = tensor.empty() : tensor<10xf32, #plan.memory_space<device>>
//   CHECK-ALLOC-DAG: %[[out2:.+]] = tensor.empty() : tensor<10xf32, #plan.memory_space<device>>
//   CHECK-ALLOC-DAG: %[[v1:.+]]:2 = linalg.generic {{.*}} ins(%[[arg0]], %[[arg1]] : {{.*}}) outs(%[[out2]], %[[out1]] : {{.*}})
//       CHECK-ALLOC: return %[[v1]]#0, %[[v1]]#1 : tensor<{{.*}}>, tensor<{{.*}}>

// -----

#map = affine_map<(d0)->(d0)>

func.func @test_no_dps_return_with_donation(%arg0: tensor<10xf32, #plan.memory_space<device>>
  {plan.aliasing_output = 0 : i32}, %arg1: tensor<10xf32, #plan.memory_space<device>>) -> tensor<10xf32, #plan.memory_space<device>> {
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

// CHECK-LABEL: @test_no_dps_return_with_donation
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32, {{.*}}> {plan.aliasing_output = 0 : i32}, %[[arg1:.+]]: tensor<10xf32, {{.*}}>)
//  CHECK-DAG: %[[out0:.+]] = tensor.empty()
//  CHECK-DAG: %[[out1:.+]] = tensor.empty()
//  CHECK-NEXT: %[[v1:.+]]:2 = {{.*}} ins(%[[arg0]], %[[arg1]] : {{.*}}) outs(%[[out1]], %[[out0]] : {{.*}})
//       CHECK: %[[v2:.+]] = arith.addf %[[v1]]#0, %[[v1]]#1 : tensor<10xf32
//  CHECK-NEXT: %[[v3:.+]] = linalg.copy  ins(%[[v2]] : {{.*}}) outs(%[[arg0]] : {{.*}})
//  CHECK-NEXT: return %[[v3]] : tensor<10xf32

// CHECK-ALLOC-LABEL: @test_no_dps_return_with_donation
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<10xf32, {{.*}}> {plan.aliasing_output = 0 : i32}, %[[arg1:.+]]: tensor<10xf32, {{.*}}>) -> tensor<10xf32, {{.*}}>
//   CHECK-ALLOC-DAG: %[[out0:.+]] = tensor.empty
//   CHECK-ALLOC-DAG: %[[out1:.+]] = tensor.empty
//   CHECK-ALLOC-DAG: %[[v1:.+]]:2 = linalg.generic {{.*}} ins(%[[arg0]], %[[arg1]] : {{.*}}) outs(%[[out1]], %[[out0]] : {{.*}})
//   CHECK-ALLOC-DAG: %[[v2:.+]] = arith.addf %[[v1]]#0, %[[v1]]#1 : tensor<10xf32
//  CHECK-ALLOC-NEXT: return %[[v2]] : tensor<10xf32

// -----

!tensor_type = tensor<10xf32, #plan.memory_space<device>>
#map = affine_map<(d0)->(d0)>
func.func @test_one_donation_other_no_dps_return(%arg0: !tensor_type {plan.aliasing_output = 1 : i32}) -> (!tensor_type, !tensor_type) {
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

//  CHECK-LABEL: @test_one_donation_other_no_dps_return
//   CHECK-SAME: (%[[arg0:.+]]: tensor<{{.*}}> {plan.aliasing_output = 1 : i32}, %[[arg1:.+]]: tensor<{{.*}}> {plan.result_arg}) -> (tensor<{{.*}}>, tensor<{{.*}}>)
//    CHECK-NOT: bufferization.alloc_tensor()
//        CHECK: %[[v0:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<{{.*}}>) outs(%[[arg0]] : tensor<{{.*}}>)
//        CHECK: %[[v1:.+]] = arith.addf %[[v0]], %[[arg0]] : tensor<{{.*}}>
//   CHECK-NEXT: %[[v2:.+]] = linalg.copy  ins(%[[v1]] : {{.*}}) outs(%[[arg1]] : {{.*}})
//   CHECK-NEXT: return %[[v2]], %[[v0]] : tensor<{{.*}}>, tensor<{{.*}}>

//  CHECK-ALLOC-LABEL: @test_one_donation_other_no_dps_return
//   CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<{{.*}}> {plan.aliasing_output = 1 : i32}) -> (tensor<{{.*}}>, tensor<{{.*}}>)
//        CHECK-ALLOC: %[[v0:.+]] = tensor.empty()
//        CHECK-ALLOC: %[[v1:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<{{.*}}>) outs(%[[v0]] : tensor<{{.*}}>)
//        CHECK-ALLOC: %[[v2:.+]] = arith.addf %[[v1]], %[[arg0]] : tensor<{{.*}}>
//   CHECK-ALLOC-NEXT: return %[[v2]], %[[v1]] : tensor<{{.*}}>, tensor<{{.*}}>

// -----

!tensor_type = tensor<10xf32, #plan.memory_space<device>>

#map = affine_map<(d0)->(d0)>
func.func @test_repeat_returns_with_donation(%arg0: !tensor_type {plan.aliasing_output = 1 : i32}, %arg1: !tensor_type) -> (!tensor_type, !tensor_type, !tensor_type, !tensor_type, !tensor_type) {
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

// CHECK-LABEL: @test_repeat_returns_with_donation
//  CHECK-SAME: (%[[arg0:.+]]: tensor<{{.*}}> {plan.aliasing_output = 1 : i32}, %[[arg1:.+]]: tensor<{{.*}}>, %[[arg2:.+]]: tensor<{{.*}}> {plan.result_arg}, %[[arg3:.+]]: tensor<{{.*}}> {plan.result_arg}, %[[arg4:.+]]: tensor<{{.*}}> {plan.result_arg}, %[[arg5:.+]]: tensor<{{.*}}> {plan.result_arg})
//   CHECK-NOT: bufferization.alloc_tensor()
//       CHECK: %[[v0:.+]]:2 = linalg.generic {{.*}} ins(%[[arg0]], %[[arg1]] : {{.*}}) outs(%[[arg2]], %[[arg0]] : {{.*}})
//       CHECK: %[[v1:.+]] = arith.addf %[[v0]]#0, %[[arg0]] : tensor<{{.*}}>
//  CHECK-NEXT: %[[v2:.+]] = linalg.copy  ins(%[[v0]]#0 : {{.*}}) outs(%[[arg3]] : {{.*}})
//  CHECK-NEXT: %[[v3:.+]] = linalg.copy  ins(%[[v0]]#1 : {{.*}}) outs(%[[arg4]] : {{.*}})
//  CHECK-NEXT: %[[v4:.+]] = linalg.copy  ins(%[[v1]] : {{.*}}) outs(%[[arg5]] : {{.*}})
//  CHECK-NEXT: return %[[v0]]#0, %[[v0]]#1, %[[v2]], %[[v3]], %[[v4]] :

// CHECK-ALLOC-LABEL: @test_repeat_returns_with_donation
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<{{.*}}> {plan.aliasing_output = 1 : i32}, %[[arg1:.+]]: tensor<{{.*}}>)
//   CHECK-ALLOC-DAG: %[[out0:.+]] = tensor.empty()
//   CHECK-ALLOC-DAG: %[[out1:.+]] = tensor.empty()
//   CHECK-ALLOC-DAG: %[[v1:.+]]:2 = linalg.generic {{.*}} ins(%[[arg0]], %[[arg1]] : {{.*}}) outs(%[[out1]], %[[out0]] : {{.*}})
//       CHECK-ALLOC: %[[v2:.+]] = arith.addf %[[v1]]#0, %[[arg0]]
//  CHECK-ALLOC-NEXT: return %[[v1]]#0, %[[v1]]#1, %[[v1]]#0, %[[v1]]#1, %[[v2]] :

// -----

!tensor_type = tensor<10xf32, #plan.memory_space<device>>

#map = affine_map<(d0)->(d0)>
func.func @test_dps_chain_repeat_with_donation(%arg0: !tensor_type {plan.aliasing_output = 0 : i32}) -> (!tensor_type, !tensor_type, !tensor_type) {
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

// CHECK-LABEL: @test_dps_chain_repeat_with_donation
//  CHECK-SAME: (%[[arg0:.+]]: tensor<{{.*}}> {plan.aliasing_output = 0 : i32}, %[[arg1:.+]]: tensor<{{.*}}> {plan.result_arg}, %[[arg2:.+]]: tensor<{{.*}}> {plan.result_arg}) -> (tensor<{{.*}}>, tensor<{{.*}}>, tensor<{{.*}}>)
//   CHECK-NOT: bufferization.alloc_tensor()
//       CHECK: %[[v0:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<{{.*}}>) outs(%[[arg1]] : tensor<{{.*}}>)
//       CHECK: %[[v1:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<{{.*}}>) outs(%[[v0]] : tensor<{{.*}}>)
//       CHECK: %[[v2:.+]] = linalg.copy  ins(%[[v1]] : {{.*}}) outs(%[[arg0]] : {{.*}})
//  CHECK-NEXT: %[[v3:.+]] = linalg.copy  ins(%[[v0]] : {{.*}}) outs(%[[arg2]] : {{.*}})
//  CHECK-NEXT: return %[[v2]], %[[v0]], %[[v3]] :

// CHECK-ALLOC-LABEL: @test_dps_chain_repeat_with_donation
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<{{.*}}> {plan.aliasing_output = 0 : i32}) -> (tensor<{{.*}}>, tensor<{{.*}}>, tensor<{{.*}}>)
//       CHECK-ALLOC: %[[v0:.+]] = tensor.empty() : tensor<10xf32, #plan.memory_space<device>>
//       CHECK-ALLOC: %[[v1:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<{{.*}}>) outs(%[[v0]] : tensor<{{.*}}>)
//       CHECK-ALLOC: %[[v2:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<{{.*}}>) outs(%[[v1]] : tensor<{{.*}}>)
//       CHECK-ALLOC: return %[[v2]], %[[v1]], %[[v1]] :

// -----

func.func @test_return_donated_arg(%arg0: tensor<10xf32> {plan.aliasing_output = 0 : i32}) -> tensor<10xf32> {
  return %arg0: tensor<10xf32>
}

// CHECK-LABEL: @test_return_donated_arg
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32> {plan.aliasing_output = 0 : i32})
//       CHECK:     %[[v0:.+]] = linalg.copy  ins(%[[arg0]] : {{.*}}) outs(%[[arg0]] : {{.*}})
//       CHECK:     return %[[v0]]

// CHECK-ALLOC-LABEL: @test_return_donated_arg
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<10xf32> {plan.aliasing_output = 0 : i32})
//  CHECK-ALLOC-NEXT:   %[[copy:.+]] = bufferization.alloc_tensor() copy(%[[arg0]])
//  CHECK-ALLOC-NEXT:   return %[[copy]]

// -----

func.func @test_donation_rewrite_for(%arg0: tensor<10xf32> {plan.aliasing_output = 0 : i32}) -> tensor<10xf32> {
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

// CHECK-LABEL: @test_donation_rewrite_for
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32> {plan.aliasing_output = 0 : i32})
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[c10:.+]] = arith.constant 10 : index
//       CHECK:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[v0:.+]] = scf.for %[[arg2:.+]] = %[[c0]] to %[[c10]] step %[[c1]] iter_args(%[[arg3:.+]] = %[[arg0]])
//       CHECK:       %[[mapped:.+]] = linalg.map { math.exp } ins(%[[arg3]] : tensor<10xf32>) outs(%[[arg3]] : tensor<10xf32>)
//       CHECK:       scf.yield %[[mapped]] : tensor<10xf32>
//       CHECK:     %[[v1:.+]] = linalg.copy  ins(%[[v0]] : {{.*}}) outs(%[[arg0]] : {{.*}})
//       CHECK:     return %[[v1]]

// CHECK-ALLOC-LABEL: @test_donation_rewrite_for
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<10xf32> {plan.aliasing_output = 0 : i32})
//       CHECK-ALLOC:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK-ALLOC:     %[[c10:.+]] = arith.constant 10 : index
//       CHECK-ALLOC:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK-ALLOC:     %[[v0:.+]] = scf.for %[[arg2:.+]] = %[[c0]] to %[[c10]] step %[[c1]] iter_args(%[[arg3:.+]] = %[[arg0]])
//       CHECK-ALLOC:       %[[mapped:.+]] = linalg.map { math.exp } ins(%[[arg3]] : tensor<10xf32>) outs(%[[arg3]] : tensor<10xf32>)
//       CHECK-ALLOC:       scf.yield %[[mapped]] : tensor<10xf32>
//   CHECK-ALLOC-DAG:     %[[copy:.+]] = bufferization.alloc_tensor() copy(%[[v0]])
//       CHECK-ALLOC:     return %[[copy]] :

// -----

func.func @test_insert_slice(%arg0: tensor<8x16x4xf32> {plan.aliasing_output = 0 : i32}) -> tensor<8x16x4xf32> {
  %t = arith.constant dense<2.0> : tensor<16x4xf32>
  %1 = tensor.insert_slice %t into %arg0[0, 0, 0][1, 16, 4][1, 1, 1] :
  tensor<16x4xf32> into tensor<8x16x4xf32>
  return %1 : tensor<8x16x4xf32>
}

// CHECK-LABEL: @test_insert_slice
//  CHECK-SAME: (%[[arg0:.+]]: tensor<8x16x4xf32> {plan.aliasing_output = 0 : i32}) -> tensor<8x16x4xf32>
//       CHECK: %[[v0:.+]] = arith.constant dense<2.000000e+00> : tensor<16x4xf32>
//  CHECK-NEXT: %[[v1:.+]] = tensor.insert_slice %[[v0]] into %[[arg0]][0, 0, 0] [1, 16, 4] [1, 1, 1] : tensor<16x4xf32> into tensor<8x16x4xf32>
//  CHECK-NEXT: %[[v2:.+]] = linalg.copy ins(%[[v1]] : tensor<8x16x4xf32>) outs(%[[arg0]] : tensor<8x16x4xf32>) -> tensor<8x16x4xf32>
//  CHECK-NEXT: return %[[v2]] : tensor<8x16x4xf32>

// CHECK-ALLOC-LABEL: @test_insert_slice
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<8x16x4xf32> {plan.aliasing_output = 0 : i32}) -> tensor<8x16x4xf32> {
//       CHECK-ALLOC: %[[v0:.+]] = arith.constant dense<2.000000e+00> : tensor<16x4xf32>
//  CHECK-ALLOC-NEXT: %[[v1:.+]] = tensor.insert_slice %[[v0]] into %[[arg0]][0, 0, 0] [1, 16, 4] [1, 1, 1] : tensor<16x4xf32> into tensor<8x16x4xf32>
//  CHECK-ALLOC-NEXT: %[[v2:.+]] = bufferization.alloc_tensor() copy(%[[v1]]) : tensor<8x16x4xf32>
//  CHECK-ALLOC-NEXT: return %[[v2]] : tensor<8x16x4xf32>