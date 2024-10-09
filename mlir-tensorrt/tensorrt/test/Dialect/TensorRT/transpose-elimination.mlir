// RUN: tensorrt-opt %s -split-input-file -tensorrt-transpose-elimination | FileCheck %s

func.func @transpose_const_fold() -> tensor<2x2xi32> {
  %const = tensorrt.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
  %1 = tensorrt.transpose {permutation = affine_map<(d0, d1)->(d1, d0)>} %const : tensor<2x2xi32> to tensor<2x2xi32>
  return %1 : tensor<2x2xi32>
}

// CHECK-LABEL: @transpose_const_fold
//       CHECK:     %[[cst_i32:.+]] = tensorrt.constant dense<{{\[}}[0, 2], [1, 3]]> : tensor<2x2xi32>
//       CHECK:     return %[[cst_i32]] : tensor<2x2xi32>

// -----

func.func @transpose_elided_const_fold() -> tensor<2x2xi32> {
  %const = tensorrt.constant dense_resource<__elided__> : tensor<2x2xi32>
  %1 = tensorrt.transpose {permutation = affine_map<(d0, d1)->(d1, d0)>} %const : tensor<2x2xi32> to tensor<2x2xi32>
  return %1 : tensor<2x2xi32>
}

// CHECK-LABEL: @transpose_elided_const_fold
//       CHECK:     %[[cst_i32:.+]] = tensorrt.constant dense_resource<__elided__> : tensor<2x2xi32>
//       CHECK:     return %[[cst_i32]] : tensor<2x2xi32>

// -----

func.func @transpose_scalar_const_fold() -> tensor<i32> {
  %const = tensorrt.constant dense<1> : tensor<i32>
  %1 = tensorrt.transpose {permutation = affine_map<()->()>} %const : tensor<i32> to tensor<i32>
  return %1 : tensor<i32>
}

// CHECK-LABEL: @transpose_scalar_const_fold
//       CHECK:     %[[cst_i32:.+]] = tensorrt.constant dense<1> : tensor<i32>
//       CHECK:     return %[[cst_i32]] : tensor<i32>

// -----

func.func @transpose_pushdown_noop(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %1 = tensorrt.transpose {permutation = affine_map<(d0, d1)->(d1, d0)>} %arg0 : tensor<2x2xf32> to tensor<2x2xf32>
  %2 = tensorrt.element_wise <kSUM> (%1, %arg1: tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-LABEL: @transpose_pushdown_noop
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x2xf32>, %[[arg1:.+]]: tensor<2x2xf32>) -> tensor<2x2xf32>
//       CHECK:     %[[v0:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[arg0]] : tensor<2x2xf32> to tensor<2x2xf32>
//       CHECK:     %[[v1:.+]] = tensorrt.element_wise <kSUM>(%[[v0]], %[[arg1]] : tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
//       CHECK:     return %[[v1]]

// -----

func.func @transpose_pushdown_switch(%arg0: tensor<2x2xf32>, %arg1: tensor<1x2xf32>) -> tensor<2x2xf32> {
  %1 = tensorrt.transpose {permutation = affine_map<(d0, d1)->(d1, d0)>} %arg0 : tensor<2x2xf32> to tensor<2x2xf32>
  %2 = tensorrt.element_wise <kSUM> (%1, %arg1: tensor<2x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-LABEL: @transpose_pushdown_switch
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x2xf32>, %[[arg1:.+]]: tensor<1x2xf32>) -> tensor<2x2xf32>
//       CHECK:     %[[v0:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[arg1]] : tensor<1x2xf32> to tensor<2x1xf32>
//       CHECK:     %[[v1:.+]] = tensorrt.element_wise <kSUM>(%[[arg0]], %[[v0]] : tensor<2x2xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
//       CHECK:     %[[v2:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[v1]] : tensor<2x2xf32> to tensor<2x2xf32>
//       CHECK:     return %[[v2]]

// -----

func.func @transpose_pushdown_transpose_elim(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %1 = tensorrt.transpose {permutation = affine_map<(d0, d1)->(d1, d0)>} %arg0 : tensor<2x2xf32> to tensor<2x2xf32>
  %2 = tensorrt.transpose {permutation = affine_map<(d0, d1)->(d1, d0)>} %arg0 : tensor<2x2xf32> to tensor<2x2xf32>
  %3 = tensorrt.element_wise <kSUM> (%1, %2: tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %3 : tensor<2x2xf32>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-LABEL: @transpose_pushdown_transpose_elim
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x2xf32>, %[[arg1:.+]]: tensor<2x2xf32>) -> tensor<2x2xf32> {
//       CHECK:     %[[v0:.+]] = tensorrt.element_wise <kSUM>(%[[arg0]], %[[arg0]] : tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
//       CHECK:     %[[v1:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[v0]] : tensor<2x2xf32> to tensor<2x2xf32>
//       CHECK:     return %[[v1]]

// -----

func.func @transpose_pushdown_cost_check(%arg0: tensor<2x2x1xf32>, %arg1: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
  %1 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2)->(d1, d0, d2)>} %arg0 : tensor<2x2x1xf32> to tensor<2x2x1xf32>
  %2 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2)->(d0, d2, d1)>} %arg1 : tensor<2x2x2xf32> to tensor<2x2x2xf32>
  %3 = tensorrt.element_wise <kSUM> (%1, %2: tensor<2x2x1xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
  return %3 : tensor<2x2x2xf32>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
//       CHECK: #[[$map1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
// CHECK-LABEL: @transpose_pushdown_cost_check
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x2x1xf32>, %[[arg1:.+]]: tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
//       CHECK:     %[[v0:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[arg0]] : tensor<2x2x1xf32> to tensor<2x1x2xf32>
//       CHECK:     %[[v1:.+]] = tensorrt.element_wise <kSUM>(%[[v0]], %[[arg1]] : tensor<2x1x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
//       CHECK:     %[[v2:.+]] = tensorrt.transpose {permutation = #[[$map1]]} %[[v1]] : tensor<2x2x2xf32> to tensor<2x2x2xf32>
//       CHECK:     return %[[v2]]

// -----

func.func @transpose_pushdown_activation(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %1 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2)->(d1, d0, d2)>} %arg0 : tensor<2x3x4xf32> to tensor<3x2x4xf32>
  %2 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %1 : tensor<3x2x4xf32>
  %3 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2)->(d1, d0, d2)>} %2 : tensor<3x2x4xf32> to tensor<2x3x4xf32>
  return %3 : tensor<2x3x4xf32>
}

// CHECK-LABEL: @transpose_pushdown_activation
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x3x4xf32>)
//       CHECK:     %[[v0:.+]] = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %[[arg0]]
//       CHECK:     return %[[v0]] :

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d3, d2, d0, d1)>

func.func @transpose_push_up_identity(%arg0: tensor<3x3x512x512xf32>) -> tensor<512x512x3x3xf16> {
  %0 = tensorrt.identity %arg0 : tensor<3x3x512x512xf32> to tensor<3x3x512x512xf16>
  %1 = tensorrt.transpose {permutation = #map} %0 : tensor<3x3x512x512xf16> to tensor<512x512x3x3xf16>
  return %1 : tensor<512x512x3x3xf16>
}

// CHECK-LABEL: @transpose_push_up_identity
//  CHECK-SAME: (%[[arg0:.+]]: tensor<3x3x512x512xf32>)
//       CHECK:     %[[v0:.+]] = tensorrt.transpose {{.*}} %[[arg0]] : tensor<3x3x512x512xf32> to tensor<512x512x3x3xf32>
//       CHECK:     %[[v1:.+]] = tensorrt.identity %[[v0]] : tensor<512x512x3x3xf32> to tensor<512x512x3x3xf16>
//       CHECK:     return %[[v1]] : tensor<512x512x3x3xf16>


// -----

#map = affine_map<(d0, d1, d2, d3) -> (d3, d2, d0, d1)>

func.func @transpose_push_up_activation() -> tensor<512x512x3x3xf32> {
  %cst_f32 = tensorrt.constant dense<0.0> : tensor<3x3x512x512xf32>
  %0 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %cst_f32 : tensor<3x3x512x512xf32>
  %1 = tensorrt.transpose {permutation = #map} %0 : tensor<3x3x512x512xf32> to tensor<512x512x3x3xf32>
  return %1 : tensor<512x512x3x3xf32>
}

// CHECK-LABEL: @transpose_push_up_activation
//       CHECK:     %[[cst_f32:.+]] = tensorrt.constant
//       CHECK:     %[[v0:.+]] = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %[[cst_f32]] : tensor<512x512x3x3xf32>
//       CHECK:     return %[[v0]] : tensor<512x512x3x3xf32>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d3, d2, d0, d1)>

func.func @transpose_push_up_unary() -> tensor<512x512x3x3xf32> {
  %cst_f32 = tensorrt.constant dense<0.0> : tensor<3x3x512x512xf32>
  %0 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kABS>} %cst_f32 : tensor<3x3x512x512xf32>
  %1 = tensorrt.transpose {permutation = #map} %0 : tensor<3x3x512x512xf32> to tensor<512x512x3x3xf32>
  return %1 : tensor<512x512x3x3xf32>
}

// CHECK-LABEL: @transpose_push_up_unary
//       CHECK:     %[[cst_f32:.+]] = tensorrt.constant
//       CHECK:     %[[v0:.+]] = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kABS>} %[[cst_f32]] : tensor<512x512x3x3xf32>
//       CHECK:     return %[[v0]] : tensor<512x512x3x3xf32>

// -----

func.func @transpose_dont_fold_multi_user_non_splat(%arg0: tensor<2x3xf32>) -> (tensor<3x2xf32>, tensor<2x3xf32>) {
  %0 = tensorrt.constant dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf32>
  %1 = tensorrt.transpose {permutation = affine_map<(d0, d1)->(d1, d0)>} %0 : tensor<2x3xf32> to tensor<3x2xf32>
  %2 = tensorrt.element_wise <kSUM>(%arg0, %0 : tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %1, %2 : tensor<3x2xf32>, tensor<2x3xf32>
}

// CHECK-LABEL: @transpose_dont_fold_multi_user_non_splat
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x3xf32>)
//       CHECK:     %[[cst_f32:.+]] = tensorrt.constant
//       CHECK:     %[[v0:.+]] = tensorrt.transpose {{.*}} %[[cst_f32]] : tensor<2x3xf32> to tensor<3x2xf32>
//       CHECK:     %[[v1:.+]] = tensorrt.element_wise <kSUM>(%[[arg0]], %[[cst_f32]] :
//       CHECK:     return %[[v0]], %[[v1]] :

// -----

func.func @push_up_transpose_elementwise_lhs(%arg0: tensor<1x197x1x64xf32>) -> tensor<16x12x197x64xf32>{
    %cst_f32 = tensorrt.constant dense<8.000000e+00> : tensor<16x197x768xf32>
    %0 = tensorrt.reshape %cst_f32 : tensor<16x197x768xf32> to tensor<16x197x12x64xf32>
    %1 = tensorrt.element_wise <kDIV>(%0, %arg0 : tensor<16x197x12x64xf32>, tensor<1x197x1x64xf32>) -> tensor<16x197x12x64xf32>
    %2 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} %1 : tensor<16x197x12x64xf32> to tensor<16x12x197x64xf32>
    return %2: tensor<16x12x197x64xf32>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
// CHECK-LABEL: @push_up_transpose_elementwise_lhs
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}})
//  CHECK-NEXT: %[[cst_f32:.+]] = tensorrt.constant
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.reshape %[[cst_f32]]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[arg0]]
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[v0]]
//  CHECK-NEXT: %[[v3:.+]] = tensorrt.element_wise <kDIV>(%[[v2]], %[[v1]] : {{.*}})
//  CHECK-NEXT: return %[[v3]]

// -----

func.func @push_up_transpose_elementwise_lhs_neg(%arg0: tensor<16x197x768xf32>, %arg1: tensor<1x1x1x1xf32>) -> tensor<16x12x197x64xf32>{
    %0 = tensorrt.reshape %arg0 : tensor<16x197x768xf32> to tensor<16x197x12x64xf32>
    %1 = tensorrt.element_wise <kDIV>(%0, %arg1 : tensor<16x197x12x64xf32>, tensor<1x1x1x1xf32>) -> tensor<16x197x12x64xf32>
    %2 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} %1 : tensor<16x197x12x64xf32> to tensor<16x12x197x64xf32>
    return %2: tensor<16x12x197x64xf32>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
// CHECK-LABEL: @push_up_transpose_elementwise_lhs_neg
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}})
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.reshape %[[arg0]]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.element_wise <kDIV>(%[[v0]], %[[arg1]] : {{.*}})
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[v1]]
//  CHECK-NEXT: return %[[v2]]

// -----

func.func @push_up_transpose_elementwise_rhs(%arg0: tensor<1x197x1x64xf32>) -> tensor<16x12x197x64xf32>{
    %cst_f32 = tensorrt.constant dense<8.000000e+00> : tensor<16x197x768xf32>
    %0 = tensorrt.reshape %cst_f32 : tensor<16x197x768xf32> to tensor<16x197x12x64xf32>
    %1 = tensorrt.element_wise <kDIV>(%arg0, %0 : tensor<1x197x1x64xf32>, tensor<16x197x12x64xf32>) -> tensor<16x197x12x64xf32>
    %2 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} %1 : tensor<16x197x12x64xf32> to tensor<16x12x197x64xf32>
    return %2: tensor<16x12x197x64xf32>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
// CHECK-LABEL: @push_up_transpose_elementwise_rhs
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}})
//  CHECK-NEXT: %[[cst_f32:.+]] = tensorrt.constant
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.reshape %[[cst_f32]]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[arg0]]
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[v0]]
//  CHECK-NEXT: %[[v3:.+]] = tensorrt.element_wise <kDIV>(%[[v1]], %[[v2]] : {{.*}})
//  CHECK-NEXT: return %[[v3]]

// -----

 func.func @push_up_transpose_elementwise_rhs_neg(%arg0: tensor<16x197x768xf32>) -> tensor<16x12x197x64xf32> {
    %cst_f32 = tensorrt.constant dense<8.000000e+00> : tensor<1x1x1x1xf32>
    %0 = tensorrt.reshape %arg0 : tensor<16x197x768xf32> to tensor<16x197x12x64xf32>
    %1 = tensorrt.element_wise <kDIV>(%cst_f32, %0 : tensor<1x1x1x1xf32>, tensor<16x197x12x64xf32>) -> tensor<16x197x12x64xf32>
    %2 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} %1 : tensor<16x197x12x64xf32> to tensor<16x12x197x64xf32>
    return %2 : tensor<16x12x197x64xf32>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
// CHECK-LABEL: @push_up_transpose_elementwise_rhs_neg
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}})
//  CHECK-NEXT: %[[cst_f32:.+]] = tensorrt.constant
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.reshape %[[arg0]]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.element_wise <kDIV>(%[[cst_f32]], %[[v0]] : {{.*}})
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[v1]]
//  CHECK-NEXT: return %[[v2]]

// -----

func.func @push_up_transpose_elementwise_negative(%arg0: tensor<10x40x30x20xf32>, %arg1: tensor<10x1x30x1xf32>) -> tensor<30x40x10x20xf32>{
    %1 = tensorrt.element_wise <kDIV>(%arg0, %arg1 : tensor<10x40x30x20xf32>, tensor<10x1x30x1xf32>) -> tensor<10x40x30x20xf32>
    %2 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>} %1 : tensor<10x40x30x20xf32> to tensor<30x40x10x20xf32>
    return %2: tensor<30x40x10x20xf32>
}

//       CHECK: #[[$map1:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>
// CHECK-LABEL: @push_up_transpose_elementwise_negative
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}})
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.element_wise <kDIV>(%[[arg0]], %[[arg1]] : {{.*}})
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.transpose {permutation = #[[$map1]]} %[[v0]]
//  CHECK-NEXT: return %[[v1]]

// -----

func.func @push_up_transpose_elementwise_negative_2(%arg0: tensor<10x20x30x40xf32>, %arg1: tensor<10x1x30x1xf32>) -> tensor<10x40x30x20xf32>{
    %0 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>} %arg0 : tensor<10x20x30x40xf32> to tensor<10x40x30x20xf32>
    %1 = tensorrt.element_wise <kDIV>(%0, %arg1 : tensor<10x40x30x20xf32>, tensor<10x1x30x1xf32>) -> tensor<10x40x30x20xf32>
    return %1: tensor<10x40x30x20xf32>
}

//       CHECK: #[[$map1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
// CHECK-LABEL: @push_up_transpose_elementwise_negative_2
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}})
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.transpose {permutation = #[[$map1]]} %[[arg1]]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.element_wise <kDIV>(%[[arg0]], %[[v0]] : {{.*}})
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.transpose {permutation = #[[$map1]]} %[[v1]]
//  CHECK-NEXT: return %[[v2]]

// -----

func.func @push_up_transpose_elementwise_rhs_constant(%arg0: tensor<1x197x1x1xf32>) -> tensor<16x12x197x64xf32>{
    %cst_f32 = tensorrt.constant dense<8.000000e+00> : tensor<16x1x12x64xf32>
    %1 = tensorrt.element_wise <kDIV>(%arg0, %cst_f32 : tensor<1x197x1x1xf32>, tensor<16x1x12x64xf32>) -> tensor<16x197x12x64xf32>
    %2 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} %1 : tensor<16x197x12x64xf32> to tensor<16x12x197x64xf32>
    return %2: tensor<16x12x197x64xf32>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
// CHECK-LABEL: @push_up_transpose_elementwise_rhs_constant
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}})
//  CHECK-NEXT: %[[cst_f32:.+]] = tensorrt.constant
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[arg0]]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.element_wise <kDIV>(%[[v0]], %[[cst_f32]] : {{.*}})
//  CHECK-NEXT: return %[[v1]]

// -----

func.func @push_up_transpose_elementwise_rhs_constant_neg(%arg0: tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>{
    %cst_f32 = tensorrt.constant dense<8.000000e+00> : tensor<1x1x1x1xf32>
    %1 = tensorrt.element_wise <kDIV>(%arg0, %cst_f32 : tensor<16x197x12x64xf32>, tensor<1x1x1x1xf32>) -> tensor<16x197x12x64xf32>
    %2 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} %1 : tensor<16x197x12x64xf32> to tensor<16x12x197x64xf32>
    return %2: tensor<16x12x197x64xf32>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
// CHECK-LABEL: @push_up_transpose_elementwise_rhs_constant_neg
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}})
//  CHECK-NEXT: %[[cst_f32:.+]] = tensorrt.constant
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.element_wise <kDIV>(%[[arg0]], %[[cst_f32]] : {{.*}})
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[v0]]
//  CHECK-NEXT: return %[[v1]]

// -----

func.func @push_up_transpose_elementwise_lhs_constant(%arg0: tensor<1x197x1x1xf32>) -> tensor<16x12x197x64xf32>{
    %cst_f32 = tensorrt.constant dense<8.000000e+00> : tensor<16x1x12x64xf32>
    %1 = tensorrt.element_wise <kDIV>(%cst_f32, %arg0 : tensor<16x1x12x64xf32>, tensor<1x197x1x1xf32>) -> tensor<16x197x12x64xf32>
    %2 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} %1 : tensor<16x197x12x64xf32> to tensor<16x12x197x64xf32>
    return %2: tensor<16x12x197x64xf32>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
// CHECK-LABEL: @push_up_transpose_elementwise_lhs_constant
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}})
//  CHECK-NEXT: %[[cst_f32:.+]] = tensorrt.constant
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[arg0]]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.element_wise <kDIV>(%[[cst_f32]], %[[v0]] : {{.*}})
//  CHECK-NEXT: return %[[v1]]

// -----

func.func @push_up_transpose_elementwise_lhs_constant_neg(%arg0: tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>{
    %cst_f32 = tensorrt.constant dense<8.000000e+00> : tensor<1x1x1x1xf32>
    %1 = tensorrt.element_wise <kDIV>(%cst_f32, %arg0 : tensor<1x1x1x1xf32>, tensor<16x197x12x64xf32>) -> tensor<16x197x12x64xf32>
    %2 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} %1 : tensor<16x197x12x64xf32> to tensor<16x12x197x64xf32>
    return %2: tensor<16x12x197x64xf32>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
// CHECK-LABEL: @push_up_transpose_elementwise_lhs_constant_neg
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}})
//  CHECK-NEXT: %[[cst_f32:.+]] = tensorrt.constant
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.element_wise <kDIV>(%[[cst_f32]], %[[arg0]] : {{.*}})
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[v0]]
//  CHECK-NEXT: return %[[v1]]

// -----

func.func @push_up_transpose_elementwise_lhs_rhs_constant() -> tensor<16x12x197x64xf32>{
    %cst_f32 = tensorrt.constant dense<8.000000e+00> : tensor<1x1x1x1xf32>
    %cst_f32_1 = tensorrt.constant dense<8.000000e+00> : tensor<16x197x12x64xf32>
    %1 = tensorrt.element_wise <kDIV>(%cst_f32, %cst_f32_1 : tensor<1x1x1x1xf32>, tensor<16x197x12x64xf32>) -> tensor<16x197x12x64xf32>
    %2 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} %1 : tensor<16x197x12x64xf32> to tensor<16x12x197x64xf32>
    return %2: tensor<16x12x197x64xf32>
}

// CHECK-LABEL: @push_up_transpose_elementwise_lhs_rhs_constant
//  CHECK-NEXT: %[[cst_f32:.+]] = tensorrt.constant
//  CHECK-NEXT: %[[cst_f32_0:.+]] = tensorrt.constant
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.element_wise <kDIV>(%[[cst_f32_0]], %[[cst_f32]] : {{.*}})
//  CHECK-NEXT: return %[[v0]]

// -----

func.func @push_up_transpose_elementwise_reshape_reshape_neg(%arg0: tensor<3152x12x64xf32>) -> tensor<16x12x197x64xf32>{
    %cst_f32 = tensorrt.constant dense<8.000000e+00> : tensor<16x197x768xf32>
    %0 = tensorrt.reshape %cst_f32 : tensor<16x197x768xf32> to tensor<16x197x12x64xf32>
    %1 = tensorrt.reshape %arg0 : tensor<3152x12x64xf32> to tensor<16x197x12x64xf32>
    %2 = tensorrt.element_wise <kDIV>(%1, %0 : tensor<16x197x12x64xf32>, tensor<16x197x12x64xf32>) -> tensor<16x197x12x64xf32>
    %3 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} %2 : tensor<16x197x12x64xf32> to tensor<16x12x197x64xf32>
    return %3: tensor<16x12x197x64xf32>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
// CHECK-LABEL: @push_up_transpose_elementwise_reshape_reshape_neg
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}})
//  CHECK-NEXT: %[[cst_f32:.+]] = tensorrt.constant
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.reshape %[[cst_f32]]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.reshape %[[arg0]]
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.element_wise <kDIV>(%[[v1]], %[[v0]] : {{.*}})
//  CHECK-NEXT: %[[v3:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[v2]]
//  CHECK-NEXT: return %[[v3]]

// -----

func.func @push_up_transpose_elementwise_transpose_transpose_neg(%arg0: tensor<10x20x30x40xf32>, %arg1: tensor<30x1x10x1xf32>) -> tensor<30x40x10x20xf32>{
    %0 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>} %arg0 : tensor<10x20x30x40xf32> to tensor<10x40x30x20xf32>
    %1 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>} %arg1 : tensor<30x1x10x1xf32> to tensor<10x1x30x1xf32>
    %2 = tensorrt.element_wise <kDIV>(%1, %0 : tensor<10x1x30x1xf32>, tensor<10x40x30x20xf32>) -> tensor<10x40x30x20xf32>
    %3 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>} %2 : tensor<10x40x30x20xf32> to tensor<30x40x10x20xf32>
    return %3: tensor<30x40x10x20xf32>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0, d1)>
// CHECK-LABEL: @push_up_transpose_elementwise_transpose_transpose_neg
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}})
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[arg1]]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.element_wise <kDIV>(%[[v0]], %[[arg0]] : {{.*}})
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[v1]]
//  CHECK-NEXT: return %[[v2]]

// -----

func.func @push_up_transpose_elementwise_reshape_transpose_neg(%arg0: tensor<10x40x600xf32>, %arg1: tensor<30x1x10x40xf32>) -> tensor<30x40x10x20xf32>{
    %0 = tensorrt.reshape %arg0 : tensor<10x40x600xf32> to tensor<10x40x30x20xf32>
    %1 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0, d1)>} %arg1 : tensor<30x1x10x40xf32> to tensor<10x40x30x1xf32>
    %2 = tensorrt.element_wise <kDIV>(%1, %0 : tensor<10x40x30x1xf32>, tensor<10x40x30x20xf32>) -> tensor<10x40x30x20xf32>
    %3 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>} %2 : tensor<10x40x30x20xf32> to tensor<30x40x10x20xf32>
    return %3: tensor<30x40x10x20xf32>
}

//   CHECK-DAG: #[[$map:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0, d1)>
//   CHECK-DAG: #[[$map1:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>
// CHECK-LABEL: @push_up_transpose_elementwise_reshape_transpose_neg
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}})
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.reshape %[[arg0]]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[arg1]]
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.element_wise <kDIV>(%[[v1]], %[[v0]] : {{.*}})
//  CHECK-NEXT: %[[v3:.+]] = tensorrt.transpose {permutation = #[[$map1]]} %[[v2]]
//  CHECK-NEXT: return %[[v3]]