// RUN: mlir-tensorrt-opt %s -split-input-file | mlir-tensorrt-opt | FileCheck %s


func.func @plan_attrs() attributes {
  plan.none_bounds = #plan.bounds<none>,
  plan.shape_bounds = #plan.bounds<shape, [1, 2, 3], [4, 5, 6]>,
  plan.value_bounds = #plan.bounds<value, dense<[1, 2, 3]> : tensor<3xi64>, dense<[4, 5, 6]> : tensor<3xi64>>
} {
  return
}

// CHECK-LABEL: @plan_attrs() attributes {
//  CHECK-SAME:   plan.none_bounds = #plan.bounds<none>
//  CHECK-SAME:   plan.shape_bounds = #plan.bounds<shape, [1, 2, 3], [4, 5, 6]>
//  CHECK-SAME:   plan.value_bounds = #plan.bounds<value, dense<[1, 2, 3]> : tensor<3xi64>, dense<[4, 5, 6]> : tensor<3xi64>>} {

// -----

func.func @plan_inline_group(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {

  %0 = plan.inline_group target(#plan.tensorrt_cluster<disallow_shape_tensor_calculations = false, benefit = 1>) -> tensor<10xf32> {
    %1 = stablehlo.add %arg0, %arg1 : tensor<10xf32>
    yield %1 : tensor<10xf32>
  }

  %1 = plan.inline_group target("some-target") -> tensor<10xf32> {
    %2 = stablehlo.add %0, %0 : tensor<10xf32>
    yield %2 : tensor<10xf32>
  }

  plan.inline_group target("some-other-target") {
    %alloc = memref.alloc() : memref<10xf32>
    memref.dealloc %alloc : memref<10xf32>
  }

  return %1 : tensor<10xf32>
}


// CHECK-LABEL: @plan_inline_group
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32>) -> tensor<10xf32> {
//       CHECK-NEXT:     %[[v0:.+]] = plan.inline_group target(#plan.tensorrt_cluster<disallow_shape_tensor_calculations = false, benefit = 1>) -> tensor<10xf32> {
//       CHECK-NEXT:       %[[v2:.+]] = stablehlo.add %[[arg0]], %[[arg1]] : tensor<10xf32>
//       CHECK-NEXT:       yield %[[v2]] : tensor<10xf32>
//       CHECK-NEXT:     }
//       CHECK-NEXT:     %[[v1:.+]] = plan.inline_group target("some-target") -> tensor<10xf32> {
//       CHECK-NEXT:       %[[v2:.+]] = stablehlo.add %[[v0]], %[[v0]] : tensor<10xf32>
//       CHECK-NEXT:       yield %[[v2]] : tensor<10xf32>
//       CHECK-NEXT:     }
//       CHECK-NEXT:     plan.inline_group target("some-other-target") {
//       CHECK-NEXT:       %[[alloc:.+]] = memref.alloc() : memref<10xf32>
//       CHECK-NEXT:       memref.dealloc %[[alloc]] : memref<10xf32>
//       CHECK-NEXT:     }
//       CHECK-NEXT:     return %[[v1]] : tensor<10xf32>

// -----

func.func @inline_closed_group(%arg0: tensor<?xf32>, %arg1: index, %arg2: tensor<?xf32>) -> tensor<?xf32> {
  %2 = plan.inline_closed_group target(#plan.tensorrt_cluster<disallow_shape_tensor_calculations = false, benefit = 1>)
    inputs(%arg0, %arg1 : tensor<?xf32>, index)
    outs(%arg2 : tensor<?xf32>)
    in_attrs [#plan.bounds<shape, [10], [20]>, #plan.bounds<none>]
    res_attrs [#plan.bounds<shape, [10], [20]>] -> tensor<?xf32> {
  ^bb0(%in0: tensor<?xf32>, %in1: index, %out0: tensor<?xf32>):
    %2 = plan.with_shape %in0 (%in1) : (tensor<?xf32>, index) -> tensor<?xf32>
    %res = stablehlo.exponential %2 : tensor<?xf32>
    yield %res : tensor<?xf32>
  }
  return %2 : tensor<?xf32>
}

// CHECK-LABEL: func.func @inline_closed_group
//       CHECK:  plan.inline_closed_group target(#plan.tensorrt_cluster<disallow_shape_tensor_calculations = false, benefit = 1>)
//  CHECK-NEXT:   inputs(%{{.+}}, %{{.+}} : tensor<?xf32>, index)
//  CHECK-NEXT:   outs(%{{.+}} : tensor<?xf32>)
//  CHECK-NEXT:   in_attrs [#plan.bounds<shape, [10], [20]>, #plan.bounds<none>]
//  CHECK-NEXT:   res_attrs [#plan.bounds<shape, [10], [20]>] -> tensor<?xf32> {
//  CHECK-NEXT:  ^bb0(%in{{.*}}: tensor<?xf32>, %in{{.*}}: index, %out{{.*}}: tensor<?xf32>):
//  CHECK-NEXT:    %{{.+}} = with_shape %in{{.*}}(%in{{.*}}) :
//  CHECK-NEXT:    %{{.+}} = stablehlo.exponential %{{.+}} : tensor<?xf32>
//  CHECK-NEXT:    yield %{{.+}} : tensor<?xf32>
//  CHECK-NEXT:  }
//  CHECK-NEXT:  return

// -----

func.func @inline_closed_alloc_group(%arg0: tensor<?xf32>, %arg1: index) -> tensor<?xf32> {
  %2 = plan.inline_closed_alloc_group target(#plan.tensorrt_cluster<disallow_shape_tensor_calculations = false, benefit = 1>)
    inputs(%arg0, %arg1 : tensor<?xf32>, index)
    in_attrs [#plan.bounds<shape, [10], [20]>, #plan.bounds<none>] -> tensor<?xf32> {
  ^bb0(%in0: tensor<?xf32>, %in1: index):
    %2 = plan.with_shape %in0 (%in1) : (tensor<?xf32>, index) -> tensor<?xf32>
    %res = stablehlo.exponential %2 : tensor<?xf32>
    yield %res : tensor<?xf32>
  }
  return %2 : tensor<?xf32>
}

// CHECK-LABEL: func.func @inline_closed_alloc_group
//       CHECK:  plan.inline_closed_alloc_group target(#plan.tensorrt_cluster<disallow_shape_tensor_calculations = false, benefit = 1>)
//  CHECK-NEXT:   inputs(%{{.+}}, %{{.+}} : tensor<?xf32>, index)
//  CHECK-NEXT:   in_attrs [#plan.bounds<shape, [10], [20]>, #plan.bounds<none>]
//  CHECK-NEXT:   -> tensor<?xf32> {
//  CHECK-NEXT:  ^bb0(%in{{.*}}: tensor<?xf32>, %in{{.*}}: index):
//  CHECK-NEXT:    %{{.+}} = with_shape %in{{.*}}(%in{{.*}}) :
//  CHECK-NEXT:    %{{.+}} = stablehlo.exponential %{{.+}} : tensor<?xf32>
//  CHECK-NEXT:    yield %{{.+}} : tensor<?xf32>
//  CHECK-NEXT:  }
//  CHECK-NEXT:  return

// -----


func.func @with_values(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  %0 = arith.constant 0 : i32
  %1 = arith.constant 1 : i32
  %2 = arith.constant 2 : i32
  %3 = arith.constant 3 : i32
  %4 = plan.with_values %arg0 (%0, %1, %2, %3) : tensor<4xi32>
  return %4 : tensor<4xi32>
}

// -----

func.func @with_values1(%arg0: tensor<0xi32>) -> tensor<0xi32> {
  %4 = plan.with_values %arg0 () : tensor<0xi32>
  return %4 : tensor<0xi32>
}

