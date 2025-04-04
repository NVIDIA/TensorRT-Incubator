// RUN: tensorrt-opt %s -tensorrt-broadcast-elimination -split-input-file | FileCheck %s

func.func @pushdown_broadcast(%arg0: tensor<1x1x10xf32>, %arg1: tensor<100x10xf32>) -> tensor<100x10xf32> {
  %0 = tensorrt.broadcast %arg0 broadcast_dims<0, 1, 2> : tensor<1x1x10xf32> to tensor<1x100x10xf32>
  %1 = tensorrt.collapse_rank %0 : tensor<1x100x10xf32> to tensor<100x10xf32>
  %2 = tensorrt.element_wise <kSUM>(%1, %arg1 : tensor<100x10xf32>, tensor<100x10xf32>) -> tensor<100x10xf32>
  return %2 : tensor<100x10xf32>
}

// CHECK-LABEL: @pushdown_broadcast
//  CHECK-NEXT:  tensorrt.collapse_rank %arg0 : tensor<1x1x10xf32> to tensor<1x10xf32>
//  CHECK-NEXT:  tensorrt.element_wise <kSUM>

// -----

func.func @pushdown_broadcast_collapse_shape_multiple_collapsed_dims() -> tensor<96x512x10x10xf32> {
  %cst_f32 = tensorrt.constant dense<1.000000e+00> : tensor<1x96x1x512x1x1xf32>
  %0 = tensorrt.broadcast %cst_f32 broadcast_dims<0, 1, 2, 3, 4, 5> : tensor<1x96x1x512x1x1xf32> to tensor<1x96x1x512x10x10xf32>
  %1 = tensorrt.collapse_rank %0 : tensor<1x96x1x512x10x10xf32> to tensor<96x512x10x10xf32>
  return %1 : tensor<96x512x10x10xf32>
}

// CHECK-LABEL: func.func @pushdown_broadcast_collapse_shape_multiple_collapsed_dims
//  CHECK-NEXT:     %[[cst_f32:.+]] = tensorrt.constant {{.*}} : tensor<96x512x1x1xf32>
//  CHECK-NEXT:     %[[v0:.+]] = tensorrt.broadcast %[[cst_f32]] broadcast_dims<0, 1, 2, 3> : tensor<96x512x1x1xf32> to tensor<96x512x10x10xf32>
//  CHECK-NEXT:     return %[[v0]] : tensor<96x512x10x10xf32>

// -----

// For this test case, the dimension removed by the collapse_rank (dim #1) is not
// part of the broadcast dimensions. Check that the 'PushDownBroadcastReduceRankOp' correctly
// exits, leaving the other patterns to simplify the IR.
func.func @pushdown_transposed_broadcast_collapse_3() -> tensor<1x96x1x1xf32> {
  %cst_f32 = tensorrt.constant dense<1.000000e+00> : tensor<1x96x1xf32>
  %0 = tensorrt.broadcast %cst_f32 broadcast_dims<0, 2, 4> : tensor<1x96x1xf32> to tensor<1x1x96x1x1xf32>
  %1 = tensorrt.collapse_rank %0 : tensor<1x1x96x1x1xf32> to tensor<1x96x1x1xf32>
  return %1 : tensor<1x96x1x1xf32>
}

// CHECK-LABEL: func.func @pushdown_transposed_broadcast_collapse_3
//  CHECK-NEXT:     %[[cst_f32:.+]] = tensorrt.constant {{.*}} : tensor<1x96x1x1xf32>
//  CHECK-NEXT:     return %[[cst_f32]] : tensor<1x96x1x1xf32>

// -----

func.func @pushdown_transposed_broadcast_collapse_4() -> tensor<96x4xf32> {
  %cst_f32 = tensorrt.constant dense<1.000000e+00> : tensor<1x96x1xf32>
  %0 = tensorrt.broadcast %cst_f32 broadcast_dims<2, 1, 0> : tensor<1x96x1xf32> to tensor<1x96x1x4xf32>
  %1 = tensorrt.collapse_rank %0 : tensor<1x96x1x4xf32> to tensor<96x4xf32>
  return %1 : tensor<96x4xf32>
}

// CHECK-LABEL: func.func @pushdown_transposed_broadcast_collapse_4
//       CHECK:  %[[cst_f32:.+]] = tensorrt.constant {{.*}} : tensor<96x1xf32>
//       CHECK:  %[[v0:.+]] = tensorrt.broadcast %[[cst_f32]] broadcast_dims<0, 1> : tensor<96x1xf32> to tensor<96x4xf32>
//       CHECK:  return %[[v0]] : tensor<96x4xf32>

// -----

func.func @pushdown_transposed_broadcast_collapse_transpose() -> tensor<96x96xf32> {
  %cst_f32 = tensorrt.constant dense<1.000000e+00> : tensor<1x96x1x96xf32>
  %0 = tensorrt.broadcast %cst_f32 broadcast_dims<2, 3, 0, 1> : tensor<1x96x1x96xf32> to tensor<1x96x1x96xf32>
  %1 = tensorrt.collapse_rank %0 : tensor<1x96x1x96xf32> to tensor<96x96xf32>
  return %1 : tensor<96x96xf32>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-LABEL: func.func @pushdown_transposed_broadcast_collapse_transpose
//  CHECK-NEXT:     %[[cst_f32:.+]] = tensorrt.constant {{.*}} : tensor<96x96xf32>
//  CHECK-NEXT:     %[[v0:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[cst_f32]] :
//  CHECK-NEXT:     return %[[v0]] : tensor<96x96xf32>

// -----

func.func @pushdown_transposed_broadcast_collapse_dynamic(%arg0: tensor<1x96x?x96xf32>) -> tensor<96x96xf32> {
  %0 = tensorrt.broadcast %arg0 broadcast_dims<2, 3, 0, 1> : tensor<1x96x?x96xf32> to tensor<1x96x1x96xf32>
  %1 = tensorrt.collapse_rank %0 : tensor<1x96x1x96xf32> to tensor<96x96xf32>
  return %1 : tensor<96x96xf32>
}

// CHECK-LABEL: func.func @pushdown_transposed_broadcast_collapse_dynamic
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x96x?x96xf32>)
//  CHECK-NEXT:   %[[v0:.+]] = tensorrt.reshape %[[arg0]] : tensor<1x96x?x96xf32> to tensor<96x96xf32>
//  CHECK-NEXT:   %[[v1:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[v0]] :
//  CHECK-NEXT:   return %[[v1]] : tensor<96x96xf32>

// -----

func.func @broadcast_ewise(%arg0: tensor<128x128xf32>, %arg1: tensor<1x128xf32>) -> tensor<128x128xf32> {
  %0 = tensorrt.broadcast %arg1 broadcast_dims<0, 1> : tensor<1x128xf32> to tensor<128x128xf32>
  %1 = tensorrt.element_wise <kSUM>(%arg0, %0 : tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
  return %1 : tensor<128x128xf32>
}

// CHECK-LABEL: @broadcast_ewise
//  CHECK-SAME:     (%[[arg0:.+]]: tensor<128x128xf32>, %[[arg1:.+]]: tensor<1x128xf32>)
//  CHECK-NEXT:   tensorrt.element_wise
//  CHECK-SAME:     <kSUM>(%[[arg0]], %[[arg1]] : tensor<128x128xf32>, tensor<1x128xf32>) -> tensor<128x128xf32>
//  CHECK-NEXT:   return

// -----

func.func @broadcast_ewise(%arg0: tensor<128x128xf32>, %arg1: tensor<1xf32>) -> tensor<128x128xf32> {
  %0 = tensorrt.broadcast %arg1 broadcast_dims<0> : tensor<1xf32> to tensor<128x128xf32>
  %1 = tensorrt.element_wise <kSUM>(%arg0, %0 : tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
  return %1 : tensor<128x128xf32>
}

// CHECK-LABEL: @broadcast_ewise
//  CHECK-SAME:     (%[[arg0:.+]]: tensor<128x128xf32>, %[[arg1:.+]]: tensor<1xf32>)
//  CHECK-NEXT:   %[[expanded:.+]] = tensorrt.expand_rank %[[arg1]] : tensor<1xf32> to tensor<1x1xf32>
//  CHECK-NEXT:   tensorrt.element_wise
//  CHECK-SAME:     <kSUM>(%[[arg0]], %[[expanded]] : tensor<128x128xf32>, tensor<1x1xf32>) -> tensor<128x128xf32>
//  CHECK-NEXT:   return

// -----

func.func @broadcast_ewise_double_one(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<128xf32> {
  %0 = tensorrt.broadcast %arg0 broadcast_dims<0> : tensor<1xf32> to tensor<128xf32>
  %1 = tensorrt.broadcast %arg1 broadcast_dims<0> : tensor<1xf32> to tensor<128xf32>
  %2 = tensorrt.element_wise <kSUM> (%0, %1 : tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
  return %2 : tensor<128xf32>
}

// CHECK-LABEL: @broadcast_ewise_double_one
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<{{.+}}>, %[[arg1:.+]]: tensor<{{.+}}>)
//       CHECK:   %[[v0:.+]] = tensorrt.broadcast %[[arg0]]
//       CHECK:   %[[v1:.+]] = tensorrt.broadcast %[[arg1]]
//       CHECK:   tensorrt.element_wise <kSUM>(%[[v0]], %[[v1]] :

// -----

func.func @broadcast_ewise_double_one_dynamic(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xi32>, %arg3: tensor<1xi32>) -> tensor<?xf32> {
  %0 = tensorrt.broadcast %arg0 broadcast_dims<0> shape(%arg2: tensor<1xi32>) : tensor<1xf32> to tensor<?xf32>
  %1 = tensorrt.broadcast %arg1 broadcast_dims<0> shape(%arg3: tensor<1xi32>) : tensor<1xf32> to tensor<?xf32>
  %2 = tensorrt.element_wise <kSUM> (%0, %1 : tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %2 : tensor<?xf32>
}

// CHECK-LABEL: @broadcast_ewise_double_one_dynamic
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<{{.+}}>, %[[arg1:.+]]: tensor<{{.+}}>, %[[arg2:.+]]: tensor<{{.+}}>, %[[arg3:.+]]: tensor<{{.+}}>)
//       CHECK:   %[[v0:.+]] = tensorrt.broadcast %[[arg0]]
//       CHECK:   %[[v1:.+]] = tensorrt.broadcast %[[arg1]]
//       CHECK:   tensorrt.element_wise <kSUM>(%[[v0]], %[[v1]] :

// -----

func.func @broadcast_ewise_dynamic_input(%arg0: tensor<1xf32>, %arg1: tensor<?xf32>, %arg2: tensor<1xi32>) -> tensor<?xf32> {
  %0 = tensorrt.broadcast %arg0 broadcast_dims<0> shape(%arg2: tensor<1xi32>) : tensor<1xf32> to tensor<?xf32>
  %2 = tensorrt.element_wise <kSUM> (%0, %arg1 : tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %2 : tensor<?xf32>
}

// CHECK-LABEL: @broadcast_ewise_dynamic_input
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<{{.+}}>, %[[arg1:.+]]: tensor<{{.+}}>, %[[arg2:.+]]: tensor<{{.+}}>)
//       CHECK:   %[[v0:.+]] = tensorrt.broadcast %[[arg0]]
//       CHECK:   tensorrt.element_wise <kSUM>(%[[v0]], %[[arg1]] :


// -----

func.func @broadcast_select(%cond: tensor<1x128xi1>, %arg0: tensor<128x128xf32>, %arg1: tensor<1xf32>) -> tensor<128x128xf32> {
  %0 = tensorrt.broadcast %arg1 broadcast_dims<0> : tensor<1xf32> to tensor<128x128xf32>
  %1 = tensorrt.broadcast %cond broadcast_dims<0, 1> : tensor<1x128xi1> to tensor<128x128xi1>
  %2 = tensorrt.select ins(%1, %arg0, %0 : tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
  return %2 : tensor<128x128xf32>
}

// CHECK-LABEL: @broadcast_select
//  CHECK-SAME:     (%[[cond:.+]]: tensor<1x128xi1>, %[[arg0:.+]]: tensor<128x128xf32>, %[[arg1:.+]]: tensor<1xf32>)
//  CHECK-NEXT:   %[[expanded:.+]] = tensorrt.expand_rank %[[arg1]] : tensor<1xf32> to tensor<1x1xf32>
//  CHECK-NEXT:   tensorrt.select
//  CHECK-SAME:     ins(%[[cond]], %[[arg0]], %[[expanded]] : tensor<1x128xi1>, tensor<128x128xf32>, tensor<1x1xf32>) -> tensor<128x128xf32>
//  CHECK-NEXT:   return

// -----

func.func @broadcast_select_negative(%cond: tensor<1x128xi1>, %arg0: tensor<1x128xf32>, %arg1: tensor<1x128xf32>) -> tensor<128x128xf32> {

  %0 = tensorrt.broadcast %cond broadcast_dims<0, 1> : tensor<1x128xi1> to tensor<128x128xi1>
  %1 = tensorrt.broadcast %arg0 broadcast_dims<0, 1> : tensor<1x128xf32> to tensor<128x128xf32>
  %2 = tensorrt.broadcast %arg1 broadcast_dims<0, 1> : tensor<1x128xf32> to tensor<128x128xf32>
  %3 = tensorrt.select ins(%0, %1, %2 : tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
  return %3 : tensor<128x128xf32>
}

// CHECK-LABEL: @broadcast_select_negative
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x128xi1>, %[[arg1:.+]]: tensor<1x128xf32>, %[[arg2:.+]]: tensor<1x128xf32>) -> tensor<128x128xf32> {
//       CHECK:     %[[v0:.+]] = tensorrt.broadcast %[[arg0]] broadcast_dims<0, 1> : tensor<1x128xi1> to tensor<128x128xi1>
//       CHECK:     %[[v1:.+]] = tensorrt.broadcast %[[arg1]] broadcast_dims<0, 1> : tensor<1x128xf32> to tensor<128x128xf32>
//       CHECK:     %[[v2:.+]] = tensorrt.broadcast %[[arg2]] broadcast_dims<0, 1> : tensor<1x128xf32> to tensor<128x128xf32>
//       CHECK:     %[[v3:.+]] = tensorrt.select ins(%[[v0]], %[[v1]], %[[v2]] :
//       CHECK:     return %[[v3]] : tensor<128x128xf32

// -----

func.func @broadcast_select_negative_dynamic(%cond: tensor<1x128xi1>, %arg0: tensor<1x128xf32>, %arg1: tensor<1x128xf32>, %arg2: tensor<2xi32>, %arg3: tensor<2xi32>, %arg4: tensor<2xi32>) -> tensor<?x?xf32> {
  %0 = tensorrt.broadcast %cond broadcast_dims<0, 1> shape(%arg2: tensor<2xi32>) : tensor<1x128xi1> to tensor<?x?xi1>
  %1 = tensorrt.broadcast %arg0 broadcast_dims<0, 1> shape(%arg3: tensor<2xi32>) : tensor<1x128xf32> to tensor<?x?xf32>
  %2 = tensorrt.broadcast %arg1 broadcast_dims<0, 1> shape(%arg4: tensor<2xi32>) : tensor<1x128xf32> to tensor<?x?xf32>
  %3 = tensorrt.select ins(%0, %1, %2 : tensor<?x?xi1>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}

// CHECK-LABEL: @broadcast_select_negative_dynamic
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x128xi1>, %[[arg1:.+]]: tensor<1x128xf32>, %[[arg2:.+]]: tensor<1x128xf32>, %[[arg3:.+]]: tensor<2xi32>, %[[arg4:.+]]: tensor<2xi32>, %[[arg5:.+]]: tensor<2xi32>) -> tensor<?x?xf32> {
//       CHECK:     %[[v0:.+]] = tensorrt.broadcast %[[arg0]] broadcast_dims<0, 1> shape(%arg3 : tensor<2xi32>) : tensor<1x128xi1> to tensor<?x?xi1>
//       CHECK:     %[[v1:.+]] = tensorrt.broadcast %[[arg1]] broadcast_dims<0, 1> shape(%arg4 : tensor<2xi32>) : tensor<1x128xf32> to tensor<?x?xf32>
//       CHECK:     %[[v2:.+]] = tensorrt.broadcast %[[arg2]] broadcast_dims<0, 1> shape(%arg5 : tensor<2xi32>) : tensor<1x128xf32> to tensor<?x?xf32>
//       CHECK:     %[[v3:.+]] = tensorrt.select ins(%[[v0]], %[[v1]], %[[v2]] :
//       CHECK:     return %[[v3]] : tensor<?x?xf32
// -----

func.func @broadcast_select_negative_dynamic_input(%cond: tensor<1x128xi1>, %arg0: tensor<1x128xf32>, %arg1: tensor<?x128xf32>, %arg2: tensor<2xi32>, %arg3: tensor<2xi32>) -> tensor<?x?xf32> {
  %0 = tensorrt.broadcast %cond broadcast_dims<0, 1> shape(%arg2: tensor<2xi32>) : tensor<1x128xi1> to tensor<?x?xi1>
  %1 = tensorrt.broadcast %arg0 broadcast_dims<0, 1> shape(%arg3: tensor<2xi32>) : tensor<1x128xf32> to tensor<?x?xf32>
  %2 = tensorrt.select ins(%0, %1, %arg1 : tensor<?x?xi1>, tensor<?x?xf32>, tensor<?x128xf32>) -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

// CHECK-LABEL: @broadcast_select_negative_dynamic_input
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x128xi1>, %[[arg1:.+]]: tensor<1x128xf32>, %[[arg2:.+]]: tensor<?x128xf32>, %[[arg3:.+]]: tensor<2xi32>, %[[arg4:.+]]: tensor<2xi32>) -> tensor<?x?xf32> {
//       CHECK:     %[[v0:.+]] = tensorrt.broadcast %[[arg0]] broadcast_dims<0, 1> shape(%arg3 : tensor<2xi32>) : tensor<1x128xi1> to tensor<?x?xi1>
//       CHECK:     %[[v1:.+]] = tensorrt.broadcast %[[arg1]] broadcast_dims<0, 1> shape(%arg4 : tensor<2xi32>) : tensor<1x128xf32> to tensor<?x?xf32>
//       CHECK:     %[[v2:.+]] = tensorrt.select ins(%[[v0]], %[[v1]], %[[arg2]] :
//       CHECK:     return %[[v2]] : tensor<?x?xf32

// -----

func.func @pushdown_broadcast(%arg0: tensor<1x10xf32>, %arg1: tensor<10x1xf32>) -> tensor<1x100x10xf32> {
  %0 = tensorrt.broadcast %arg0 broadcast_dims<1, 2> : tensor<1x10xf32> to tensor<1x100x10xf32>
  %1 = tensorrt.broadcast %arg1 broadcast_dims<2, 1> : tensor<10x1xf32> to tensor<1x100x10xf32>
  %2 = tensorrt.element_wise <kSUM>(%0, %1 : tensor<1x100x10xf32>, tensor<1x100x10xf32>) -> tensor<1x100x10xf32>
  return %2 : tensor<1x100x10xf32>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-LABEL: @pushdown_broadcast
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x10xf32>, %[[arg1:.+]]: tensor<10x1xf32>) -> tensor<1x100x10xf32> {
//       CHECK:     %[[v0:.+]] = tensorrt.expand_rank %[[arg0]] : tensor<1x10xf32> to tensor<1x1x10xf32>
//       CHECK:     %[[v1:.+]] = tensorrt.broadcast %[[v0]] broadcast_dims<0, 1, 2> : tensor<1x1x10xf32> to tensor<1x100x10xf32>
//       CHECK:     %[[v2:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[arg1]] : tensor<10x1xf32> to tensor<1x10xf32>
//       CHECK:     %[[v3:.+]] = tensorrt.expand_rank %[[v2]] : tensor<1x10xf32> to tensor<1x1x10xf32>
//       CHECK:     %[[v4:.+]] = tensorrt.element_wise <kSUM>(%[[v1]], %[[v3]] : tensor<1x100x10xf32>, tensor<1x1x10xf32>) -> tensor<1x100x10xf32>
//       CHECK:     return %[[v4]] : tensor<1x100x10xf32

// -----

func.func @simplify_broadcast_dynamic(%arg0: tensor<?xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %1 = tensorrt.broadcast %arg0 broadcast_dims<0> shape(%arg1 : tensor<2xi32>) : tensor<?xf32> to tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @simplify_broadcast_dynamic
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xf32>, %[[arg1:.+]]: tensor<2xi32>)
//  CHECK-NEXT:     %[[v0:.+]] = tensorrt.expand_rank %[[arg0]] : tensor<?xf32> to tensor<?x1xf32>
//  CHECK-NEXT:     %[[v1:.+]] = tensorrt.broadcast %[[v0]] broadcast_dims<0, 1> shape(%[[arg1]] : tensor<2xi32>) : tensor<?x1xf32> to tensor<?x?xf32>
//  CHECK-NEXT:     return %[[v1]] :

// -----

func.func @simplify_broadcast_scalar_dynamic(%arg0: tensor<f32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %1 = tensorrt.broadcast %arg0 broadcast_dims<> shape(%arg1 : tensor<2xi32>) : tensor<f32> to tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @simplify_broadcast_scalar_dynamic
//  CHECK-SAME: (%[[arg0:.+]]: tensor<f32>, %[[arg1:.+]]: tensor<2xi32>)
//  CHECK-NEXT:     %[[v0:.+]] = tensorrt.expand_rank %[[arg0]] : tensor<f32> to tensor<1x1xf32>
//  CHECK-NEXT:     %[[v1:.+]] = tensorrt.broadcast %[[v0]] broadcast_dims<0, 1> shape(%[[arg1]] : tensor<2xi32>) : tensor<1x1xf32> to tensor<?x?xf32>
//  CHECK-NEXT:     return %[[v1]] :


// -----

func.func @broadcast_elim_matmul_rhs_only(%arg0: tensor<?x?x?x128xf32>, %arg1: tensor<?x?x100x128xf32>,
              %lhs_shape: tensor<4xi32>, %rhs_shape: tensor<4xi32>) -> tensor<?x?x100x100xf32> {
  %lhs = tensorrt.broadcast %arg0 broadcast_dims <0, 1, 2, 3> shape(%lhs_shape: tensor<4xi32>) : tensor<?x?x?x128xf32> to tensor<?x?x100x128xf32>
  %rhs = tensorrt.broadcast %arg1 broadcast_dims <0, 1, 2, 3> shape(%rhs_shape: tensor<4xi32>) : tensor<?x?x100x128xf32> to tensor<?x?x100x128xf32>
  %1 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kTRANSPOSE> }
    ins(%lhs, %rhs : tensor<?x?x100x128xf32>, tensor<?x?x100x128xf32>) -> tensor<?x?x100x100xf32>
  return %1 : tensor<?x?x100x100xf32>
}

// CHECK-LABEL: func.func @broadcast_elim_matmul_rhs_only
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?x?x128xf32>, %[[arg1:.+]]: tensor<?x?x100x128xf32>, %[[arg2:.+]]: tensor<4xi32>, %[[arg3:.+]]: tensor<4xi32>) -> tensor<?x?x100x100xf32> {
//       CHECK:     %[[v0:.+]] = tensorrt.broadcast %[[arg0]] broadcast_dims<0, 1, 2, 3> shape(%[[arg2]] : tensor<4xi32>) : tensor<?x?x?x128xf32> to tensor<?x?x100x128xf32>
//       CHECK:     %[[v1:.+]] = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kTRANSPOSE>} ins(%[[v0]], %[[arg1]] : tensor<?x?x100x128xf32>, tensor<?x?x100x128xf32>) -> tensor<?x?x100x100xf32>
//       CHECK:     return %[[v1]] : tensor<?x?x100x100xf32>

// -----

func.func @broadcast_elim_matmul_lhs_only(%arg0: tensor<?x1x100x128xf32>, %arg1: tensor<1x?x100x?xf32>,
              %lhs_shape: tensor<4xi32>, %rhs_shape: tensor<4xi32>) -> tensor<?x?x100x100xf32> {
  %lhs = tensorrt.broadcast %arg0 broadcast_dims <0, 1, 2, 3> shape(%lhs_shape: tensor<4xi32>) : tensor<?x1x100x128xf32> to tensor<?x?x100x128xf32>
  %rhs = tensorrt.broadcast %arg1 broadcast_dims <0, 1, 2, 3> shape(%rhs_shape: tensor<4xi32>) : tensor<1x?x100x?xf32> to tensor<?x?x100x128xf32>
  %1 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kTRANSPOSE> }
    ins(%lhs, %rhs : tensor<?x?x100x128xf32>, tensor<?x?x100x128xf32>) -> tensor<?x?x100x100xf32>
  return %1 : tensor<?x?x100x100xf32>
}

// CHECK-LABEL: func.func @broadcast_elim_matmul_lhs_only
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x1x100x128xf32>, %[[arg1:.+]]: tensor<1x?x100x?xf32>, %[[arg2:.+]]: tensor<4xi32>, %[[arg3:.+]]: tensor<4xi32>) -> tensor<?x?x100x100xf32> {
//       CHECK:     %[[v0:.+]] = tensorrt.broadcast %[[arg1]] broadcast_dims<0, 1, 2, 3> shape(%[[arg3]] : tensor<4xi32>) : tensor<1x?x100x?xf32> to tensor<?x?x100x128xf32>
//       CHECK:     %[[v1:.+]] = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kTRANSPOSE>} ins(%[[arg0]], %[[v0]] : tensor<?x1x100x128xf32>, tensor<?x?x100x128xf32>) -> tensor<?x?x100x100xf32>
//       CHECK:     return %[[v1]] : tensor<?x?x100x100xf32>

// -----

func.func @broadcast_elim_matmul_simple(%arg0: tensor<?x?x100x128xf32>, %arg1: tensor<?x?x100x128xf32>,
              %lhs_shape: tensor<4xi32>, %rhs_shape: tensor<4xi32>) -> tensor<?x?x100x100xf32> {
  %lhs = tensorrt.broadcast %arg0 broadcast_dims <0, 1, 2, 3> shape(%lhs_shape: tensor<4xi32>) : tensor<?x?x100x128xf32> to tensor<?x?x100x128xf32>
  %rhs = tensorrt.broadcast %arg1 broadcast_dims <0, 1, 2, 3> shape(%rhs_shape: tensor<4xi32>) : tensor<?x?x100x128xf32> to tensor<?x?x100x128xf32>
  %1 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kTRANSPOSE> }
    ins(%lhs, %rhs : tensor<?x?x100x128xf32>, tensor<?x?x100x128xf32>) -> tensor<?x?x100x100xf32>
  return %1 : tensor<?x?x100x100xf32>
}

// CHECK-LABEL: func.func @broadcast_elim_matmul_simple
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?x100x128xf32>, %[[arg1:.+]]: tensor<?x?x100x128xf32>, %[[arg2:.+]]: tensor<4xi32>, %[[arg3:.+]]: tensor<4xi32>) -> tensor<?x?x100x100xf32> {
//       CHECK:     %[[v0:.+]] = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kTRANSPOSE>} ins(%[[arg0]], %[[arg1]] : tensor<?x?x100x128xf32>, tensor<?x?x100x128xf32>) -> tensor<?x?x100x100xf32>
//       CHECK:     return %[[v0]] : tensor<?x?x100x100xf32>


// -----

func.func @broadcast_elim_matmul_simple_1_batch_dim(%arg0: tensor<?x100x128xf32>, %arg1: tensor<?x100x128xf32>,
              %lhs_shape: tensor<3xi32>, %rhs_shape: tensor<3xi32>) -> tensor<?x100x100xf32> {
  %lhs = tensorrt.broadcast %arg0 broadcast_dims <0, 1, 2> shape(%lhs_shape: tensor<3xi32>) : tensor<?x100x128xf32> to tensor<?x100x128xf32>
  %rhs = tensorrt.broadcast %arg1 broadcast_dims <0, 1, 2> shape(%rhs_shape: tensor<3xi32>) : tensor<?x100x128xf32> to tensor<?x100x128xf32>
  %1 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kTRANSPOSE> }
    ins(%lhs, %rhs : tensor<?x100x128xf32>, tensor<?x100x128xf32>) -> tensor<?x100x100xf32>
  return %1 : tensor<?x100x100xf32>
}

// CHECK-LABEL: func.func @broadcast_elim_matmul_simple_1_batch_dim
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x100x128xf32>, %[[arg1:.+]]: tensor<?x100x128xf32>, %[[arg2:.+]]: tensor<3xi32>, %[[arg3:.+]]: tensor<3xi32>) -> tensor<?x100x100xf32> {
//       CHECK:     %[[v0:.+]] = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kTRANSPOSE>} ins(%[[arg0]], %[[arg1]] : tensor<?x100x128xf32>, tensor<?x100x128xf32>) -> tensor<?x100x100xf32>
//       CHECK:     return %[[v0]] : tensor<?x100x100xf32>

// -----

func.func @broadcast_elim_matmul_simple_1_batch_dim_one_broadcast(%arg0: tensor<?x100x128xf32>, %arg1: tensor<?x100x128xf32>,
              %lhs_shape: tensor<3xi32>) -> tensor<?x100x100xf32> {
  %lhs = tensorrt.broadcast %arg0 broadcast_dims <0, 1, 2> shape(%lhs_shape: tensor<3xi32>) : tensor<?x100x128xf32> to tensor<?x100x128xf32>
  %1 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kTRANSPOSE> }
    ins(%lhs, %arg1 : tensor<?x100x128xf32>, tensor<?x100x128xf32>) -> tensor<?x100x100xf32>
  return %1 : tensor<?x100x100xf32>
}

// CHECK-LABEL: func.func @broadcast_elim_matmul_simple_1_batch_dim_one_broadcast
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x100x128xf32>, %[[arg1:.+]]: tensor<?x100x128xf32>, %[[arg2:.+]]: tensor<3xi32>) -> tensor<?x100x100xf32> {
//       CHECK:     %[[v0:.+]] = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kTRANSPOSE>} ins(%[[arg0]], %[[arg1]] : tensor<?x100x128xf32>, tensor<?x100x128xf32>) -> tensor<?x100x100xf32>
//       CHECK:     return %[[v0]] : tensor<?x100x100xf32>

// -----

func.func @broadcast_elim_matmul_vector(%arg0: tensor<?x?x128xf32>, %arg1: tensor<?x?x100x128xf32>,
              %lhs_shape: tensor<3xi32>, %rhs_shape: tensor<4xi32>) -> tensor<?x?x100xf32> {
  %lhs = tensorrt.broadcast %arg0 broadcast_dims <0, 1, 2> shape(%lhs_shape: tensor<3xi32>) : tensor<?x?x128xf32> to tensor<?x?x128xf32>
  %rhs = tensorrt.broadcast %arg1 broadcast_dims <0, 1, 2, 3> shape(%rhs_shape: tensor<4xi32>) : tensor<?x?x100x128xf32> to tensor<?x?x100x128xf32>
  %1 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kVECTOR>, op1 = #tensorrt.matrix_operation<kTRANSPOSE> }
    ins(%lhs, %rhs : tensor<?x?x128xf32>, tensor<?x?x100x128xf32>) -> tensor<?x?x100xf32>
  return %1 : tensor<?x?x100xf32>
}

// CHECK-LABEL: func.func @broadcast_elim_matmul_vector
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?x128xf32>, %[[arg1:.+]]: tensor<?x?x100x128xf32>, %[[arg2:.+]]: tensor<3xi32>, %[[arg3:.+]]: tensor<4xi32>) -> tensor<?x?x100xf32> {
//       CHECK:     %[[v0:.+]] = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kVECTOR>, op1 = #tensorrt.matrix_operation<kTRANSPOSE>} ins(%[[arg0]], %[[arg1]] : tensor<?x?x128xf32>, tensor<?x?x100x128xf32>) -> tensor<?x?x100xf32>
//       CHECK:     return %[[v0]] : tensor<?x?x100xf32>


// -----

func.func @broadcast_dynamic_expand_shape_regression(%arg0: tensor<?x1xf16>,  %arg1: tensor<4xi32>) -> tensor<?x?x256x256xf16> {
  %1 = tensorrt.broadcast %arg0 broadcast_dims<2, 3> shape(%arg1 : tensor<4xi32>) : tensor<?x1xf16> to tensor<?x?x256x256xf16>
  return %1 : tensor<?x?x256x256xf16>
}

// CHECK-LABEL: func.func @broadcast_dynamic_expand_shape_regression
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x1xf16>, %[[arg1:.+]]: tensor<4xi32>) -> tensor<?x?x256x256xf16> {
//       CHECK:     %[[v0:.+]] = tensorrt.reshape %[[arg0]] : tensor<?x1xf16> to tensor<1x1x?x1xf16>

// -----

func.func @broadcast_dynamic_expand_shape_regression(%arg1: tensor<?x1x?xf16>, %arg3: tensor<4xi32>) -> tensor<?x?x256x256xf16> {
  %1 = tensorrt.broadcast %arg1 broadcast_dims<3, 2, 1> shape(%arg3 : tensor<4xi32>) : tensor<?x1x?xf16> to tensor<?x?x256x256xf16>
  return %1 : tensor<?x?x256x256xf16>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2) -> (d2, d1, d0)>
// CHECK-LABEL: func.func @broadcast_dynamic_expand_shape_regression
//  CHECK-SAME: (%[[arg1:.+]]: tensor<?x1x?xf16>, %[[arg3:.+]]: tensor<4xi32>) -> tensor<?x?x256x256xf16> {
//       CHECK:     %[[cst_i32:.+]] = tensorrt.constant dense<1> : tensor<1xi32>
//       CHECK:     %[[v0:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[arg1]] : tensor<?x1x?xf16> to tensor<?x1x?xf16>
//       CHECK:     %[[v1:.+]] = tensorrt.shape %[[v0]] : tensor<?x1x?xf16> -> tensor<3xi32>
//       CHECK:     %[[v2:.+]] = tensorrt.slice %[[v1]][0][1][1] : tensor<3xi32> to tensor<1xi32>
//       CHECK:     %[[v3:.+]] = tensorrt.slice %[[v1]][2][1][1] : tensor<3xi32> to tensor<1xi32>
//       CHECK:     %[[v4:.+]] = tensorrt.concatenation {axis = 0 : i32} ins(%[[cst_i32]], %[[v2]], %[[cst_i32]], %[[v3]] : tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
//       CHECK:     %[[v5:.+]] = tensorrt.reshape %[[v0]] shape(%[[v4]]: tensor<4xi32>) : tensor<?x1x?xf16> to tensor<1x?x1x?xf16>
