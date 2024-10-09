// RUN: mlir-tensorrt-opt -split-input-file %s --convert-stablehlo-to-tensorrt -allow-unregistered-dialect | FileCheck %s

func.func @hlo_add_f32_static(%lhs: tensor<128x128xf32>, %rhs: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = "stablehlo.add"(%lhs, %rhs) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-LABEL: @hlo_add_f32_static
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<{{.+}}>, %[[arg1:.+]]: tensor<{{.+}}>)
//       CHECK:    tensorrt.element_wise
//  CHECK-SAME:      <kSUM>

// -----

func.func @hlo_sub_f32(%lhs: tensor<?x128xf32>, %rhs: tensor<?x128xf32>) -> tensor<?x128xf32> {
  %0 = "stablehlo.subtract"(%lhs, %rhs) : (tensor<?x128xf32>, tensor<?x128xf32>) -> tensor<?x128xf32>
  return %0 : tensor<?x128xf32>
}
// CHECK-LABEL: @hlo_sub_f32
//       CHECK:    tensorrt.element_wise
//  CHECK-SAME:      <kSUB>

// -----

func.func @hlo_multiply_f32(%lhs: tensor<?x128xf32>, %rhs: tensor<?x128xf32>) -> tensor<?x128xf32> {
  %0 = "stablehlo.multiply"(%lhs, %rhs) : (tensor<?x128xf32>, tensor<?x128xf32>) -> tensor<?x128xf32>
  return %0 : tensor<?x128xf32>
}
// CHECK-LABEL: @hlo_multiply_f32
//       CHECK:    tensorrt.element_wise
//  CHECK-SAME:      <kPROD>

// -----

func.func @hlo_divide_f32(%lhs: tensor<?x128xf32>, %rhs: tensor<?x128xf32>) -> tensor<?x128xf32> {
  %0 = "stablehlo.divide"(%lhs, %rhs) : (tensor<?x128xf32>, tensor<?x128xf32>) -> tensor<?x128xf32>
  return %0 : tensor<?x128xf32>
}
// CHECK-LABEL: @hlo_divide_f32
//       CHECK:    tensorrt.element_wise
//  CHECK-SAME:      <kDIV>

// -----

func.func @hlo_power(%lhs: tensor<?x128xf32>, %rhs: tensor<?x128xf32>) -> tensor<?x128xf32> {
  %0 = "stablehlo.power"(%lhs, %rhs) : (tensor<?x128xf32>, tensor<?x128xf32>) -> tensor<?x128xf32>
  return %0 : tensor<?x128xf32>
}
// CHECK-LABEL: @hlo_power
//       CHECK:    tensorrt.element_wise
//  CHECK-SAME:      <kPOW>

// -----

func.func @hlo_xor(%lhs: tensor<2xi1>, %rhs: tensor<2xi1>) -> tensor<2xi1> {
  %0 = "stablehlo.xor"(%lhs, %rhs) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}

// CHECK-LABEL: @hlo_xor
//       CHECK:    tensorrt.element_wise
//  CHECK-SAME:      <kXOR>

// -----

func.func @hlo_xor_unsupported(%lhs: tensor<2xi32>, %rhs: tensor<2xi32>) -> tensor<2xi32> {
  %0 = "stablehlo.xor"(%lhs, %rhs) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// CHECK-LABEL: @hlo_xor_unsupported
//   CHECK-NOT:  tensorrt
//       CHECK:  stablehlo.xor

// -----

func.func @hlo_constant() -> tensor<1x128x64xf32> {
  %0 = "stablehlo.constant"() {value = dense<1.0> : tensor<1x128x64xf32>} : () -> tensor<1x128x64xf32>
  return %0 : tensor<1x128x64xf32>
}

// CHECK-LABEL: @hlo_constant
//       CHECK:   tensorrt.constant
//  CHECK-SAME:       dense<1.0{{.+}}> : tensor<1x128x64xf32>

// -----

func.func @hlo_constant_bool() -> tensor<1x128x64xi1> {
  %0 = "stablehlo.constant"() {value = dense<true> : tensor<1x128x64xi1>} : () -> tensor<1x128x64xi1>
  return %0 : tensor<1x128x64xi1>
}

// CHECK-LABEL: @hlo_constant_bool
//       CHECK:   tensorrt.constant
//  CHECK-SAME:     : tensor<1x128x64xi32>
//       CHECK:   tensorrt.identity
//  CHECK-SAME:     to tensor<1x128x64xi1>

// -----

func.func @hlo_abs(%arg0: tensor<1x128x64xf32>) -> tensor<1x128x64xf32> {
  %0 = "stablehlo.abs"(%arg0) : (tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
  return %0 : tensor<1x128x64xf32>
}

// CHECK-LABEL: @hlo_abs
//       CHECK:   tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kABS>}

// -----

func.func @hlo_ceil(%arg0: tensor<1x128x64xf32>) -> tensor<1x128x64xf32> {
  %0 = "stablehlo.ceil"(%arg0) : (tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
  return %0 : tensor<1x128x64xf32>
}

// CHECK-LABEL: @hlo_ceil
//       CHECK:   tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kCEIL>}

// -----

func.func @hlo_floor(%arg0: tensor<1x128x64xf32>) -> tensor<1x128x64xf32> {
  %0 = "stablehlo.floor"(%arg0) : (tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
  return %0 : tensor<1x128x64xf32>
}

// CHECK-LABEL: @hlo_floor
//       CHECK:   tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kFLOOR>}


// -----

func.func @hlo_exponential(%arg0: tensor<1x128x64xf32>) -> tensor<1x128x64xf32> {
  %0 = "stablehlo.exponential"(%arg0) : (tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
  return %0 : tensor<1x128x64xf32>
}

// CHECK-LABEL: @hlo_exponential
//       CHECK:   tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kEXP>}

// -----

func.func @hlo_cosine(%arg0: tensor<1x128x64xf32>) -> tensor<1x128x64xf32> {
  %0 = "stablehlo.cosine"(%arg0) : (tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
  return %0 : tensor<1x128x64xf32>
}

// CHECK-LABEL: @hlo_cosine
//       CHECK:   tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kCOS>}

// -----

func.func @hlo_sine(%arg0: tensor<1x128x64xf32>) -> tensor<1x128x64xf32> {
  %0 = "stablehlo.sine"(%arg0) : (tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
  return %0 : tensor<1x128x64xf32>
}

// CHECK-LABEL: @hlo_sine
//       CHECK:   tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kSIN>}

// -----

func.func @hlo_sqrt(%arg0: tensor<1x128x64xf32>) -> tensor<1x128x64xf32> {
  %0 = "stablehlo.sqrt"(%arg0) : (tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
  return %0 : tensor<1x128x64xf32>
}

// CHECK-LABEL: @hlo_sqrt
//       CHECK:   tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kSQRT>}

// -----

func.func @hlo_negate(%arg0: tensor<1x128x64xf32>) -> tensor<1x128x64xf32> {
  %0 = "stablehlo.negate"(%arg0) : (tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
  return %0 : tensor<1x128x64xf32>
}

// CHECK-LABEL: @hlo_negate
//       CHECK:   tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kNEG>}

// -----

func.func @hlo_negate_i32(%arg0: tensor<1x128x64xi32>) -> tensor<1x128x64xi32> {
  %0 = "stablehlo.negate"(%arg0) : (tensor<1x128x64xi32>) -> tensor<1x128x64xi32>
  return %0 : tensor<1x128x64xi32>
}

// CHECK-LABEL: @hlo_negate_i32
//  CHECK-SAME:   (%[[arg0:.+]]: tensor<{{.+}})
//       CHECK:   %[[i32:.+]] = tensorrt.identity %[[arg0]] : tensor<1x128x64xi32> to tensor<1x128x64xf32>
//       CHECK:   %[[un:.+]] = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kNEG>} %[[i32]]
//       CHECK:   tensorrt.identity %[[un]] : tensor<1x128x64xf32> to tensor<1x128x64xi32>

// -----

func.func @hlo_negate_i32_scalar(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "stablehlo.negate"(%arg0) : (tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: @hlo_negate_i32_scalar
//       CHECK: %[[exp:.+]] = tensorrt.expand_rank %[[arg0:.+]] : tensor<i32> to tensor<1xi32>
//       CHECK: %[[f32:.+]] = tensorrt.identity %[[exp]] : tensor<1xi32> to tensor<1xf32>
//       CHECK: %[[un:.+]] = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kNEG>} %[[f32]] : tensor<1xf32>
//       CHECK: %[[i32:.+]] = tensorrt.identity %[[un]] : tensor<1xf32> to tensor<1xi32>
//       CHECK: %[[coll:.+]] = tensorrt.collapse_rank %[[i32]] : tensor<1xi32> to tensor<i32>
// -----

func.func @hlo_negate_f32_scalar(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.negate"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: @hlo_negate_f32_scalar
//       CHECK: %[[exp:.+]] = tensorrt.expand_rank %[[arg0:.+]] : tensor<f32> to tensor<1xf32>
//       CHECK: %[[un:.+]] = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kNEG>} %[[exp]] : tensor<1xf32>
//       CHECK: %[[coll:.+]] = tensorrt.collapse_rank %[[un]] : tensor<1xf32> to tensor<f32>

// -----

func.func @hlo_log(%arg0: tensor<1x128x64xf32>) -> tensor<1x128x64xf32> {
  %0 = "stablehlo.log"(%arg0) : (tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
  return %0 : tensor<1x128x64xf32>
}

// CHECK-LABEL: @hlo_log
//       CHECK:   tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kLOG>}

// -----

func.func @hlo_not(%arg0: tensor<1x128x64xi1>) -> tensor<1x128x64xi1> {
  %0 = "stablehlo.not"(%arg0) : (tensor<1x128x64xi1>) -> tensor<1x128x64xi1>
  return %0 : tensor<1x128x64xi1>
}

// CHECK-LABEL: @hlo_not
//       CHECK:   tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kNOT>}

// -----

func.func @op_slice(%arg0: tensor<16xf32>) -> tensor<4xf32> {
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: 0>,
    limit_indices = array<i64: 4>,
    strides = array<i64: 1>
  } : (tensor<16xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// CHECK-LABEL: @op_slice
//  CHECK-SAME:   (%[[arg0:.+]]: tensor<{{.+}}>)
//  CHECK-NEXT:   tensorrt.slice %[[arg0]][0][4][1]

// -----

func.func @op_slice_non_unit_stride(%arg0: tensor<16xf32>) -> tensor<4xf32> {
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: 8>,
    limit_indices = array<i64: 16>,
    strides = array<i64: 2>
  } : (tensor<16xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// CHECK-LABEL: @op_slice_non_unit_stride
//  CHECK-SAME:   (%[[arg0:.+]]: tensor<{{.+}}>)
//  CHECK-NEXT:   tensorrt.slice %[[arg0]][8][4][2]

// -----

func.func @hlo_dot(%arg0: tensor<10x20xi32>, %arg1: tensor<20x30xi32>) -> tensor<10x30xi32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) {} : (tensor<10x20xi32>, tensor<20x30xi32>) -> tensor<10x30xi32>
  return %0 : tensor<10x30xi32>
}

// CHECK-LABEL: @hlo_dot
//  CHECK-NEXT: %[[i0:.+]] = tensorrt.identity %[[arg0:.+]] : tensor<10x20xi32> to tensor<10x20xf32>
//  CHECK-NEXT: %[[i1:.+]] = tensorrt.identity %[[arg1:.+]] : tensor<20x30xi32> to tensor<20x30xf32>
//  CHECK-NEXT: %[[out:.+]] = tensorrt.matrix_multiply {
//  CHECK-SAME: op0 = #tensorrt.matrix_operation<kNONE>,
//  CHECK-SAME: op1 = #tensorrt.matrix_operation<kNONE>}
//  CHECK-SAME: ins(%[[i0]], %[[i1]] : tensor<10x20xf32>, tensor<20x30xf32>) -> tensor<10x30xf32>
//  CHECK-NEXT: %[[i3:.+]] = tensorrt.identity %[[out]] : tensor<10x30xf32> to tensor<10x30xi32>


// -----

func.func @hlo_dot(%arg0: tensor<10x20xf16>, %arg1: tensor<20x30xf16>) -> tensor<10x30xf16> {
  %0 = "stablehlo.dot"(%arg0, %arg1) {} : (tensor<10x20xf16>, tensor<20x30xf16>) -> tensor<10x30xf16>
  return %0 : tensor<10x30xf16>
}

// CHECK-LABEL: @hlo_dot
//  CHECK-NEXT:  tensorrt.matrix_multiply {
//  CHECK-SAME:   op0 = #tensorrt.matrix_operation<kNONE>,
//  CHECK-SAME:   op1 = #tensorrt.matrix_operation<kNONE>}
//  CHECK-SAME:    ins(%{{.+}}, %{{.+}} : tensor<10x20xf16>, tensor<20x30xf16>) -> tensor<10x30xf16>

// -----

func.func @hlo_dot(%arg0: tensor<10x20xf32>, %arg1: tensor<20xf32>) -> tensor<10xf32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) {} : (tensor<10x20xf32>, tensor<20xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @hlo_dot
//  CHECK-NEXT:  tensorrt.matrix_multiply {
//  CHECK-SAME:   op0 = #tensorrt.matrix_operation<kNONE>,
//  CHECK-SAME:   op1 = #tensorrt.matrix_operation<kVECTOR>}
//  CHECK-SAME:    ins(%{{.+}}, %{{.+}} : tensor<10x20xf32>, tensor<20xf32>) -> tensor<10xf32>

// -----

func.func @hlo_reshape(%arg0: tensor<10xf32>) -> tensor<2x5xf32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<10xf32>) -> tensor<2x5xf32>
  return %0 : tensor<2x5xf32>
}

// CHECK-LABEL: @hlo_reshape
//       CHECK:  tensorrt.reshape %{{.+}} : tensor<10xf32> to tensor<2x5xf32>

// -----

func.func @hlo_reshape_collapse_rank(%arg0: tensor<10x1xf32>) -> tensor<10xf32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<10x1xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @hlo_reshape_collapse_rank
//       CHECK:  tensorrt.collapse_rank

// -----

func.func @hlo_dynamic_reshape(%arg0: tensor<?xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %0 = "stablehlo.dynamic_reshape"(%arg0, %arg1) : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @hlo_dynamic_reshape
//  CHECK-SAME:    (%[[arg0:.+]]: tensor<{{.+}}>, %[[arg1:.+]]: tensor<{{.+}}>)
//       CHECK:  tensorrt.reshape %[[arg0]] shape(%[[arg1]]: tensor<2xi32>) : tensor<?xf32> to tensor<?x?xf32>


// -----

func.func @hlo_reshape_expand_rank(%arg0: tensor<10xf32>) -> tensor<10x1xf32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<10xf32>) -> tensor<10x1xf32>
  return %0 : tensor<10x1xf32>
}

// CHECK-LABEL: @hlo_reshape_expand_rank
//       CHECK:  tensorrt.expand_rank

// -----

func.func @hlo_broadcast_in_dim(%arg0: tensor<10xf32>) -> tensor<1x10x10x1xf32> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions=array<i64: 2>}: (tensor<10xf32>) -> tensor<1x10x10x1xf32>
  return %0 : tensor<1x10x10x1xf32>
}

// CHECK-LABEL: @hlo_broadcast_in_dim
//  CHECK-NEXT:   %0 = tensorrt.broadcast %{{.+}} broadcast_dims<2> :
//  CHECK-SAME:     tensor<10xf32> to tensor<1x10x10x1xf32>

// -----

func.func @hlo_dynamic_broadcast_in_dim_static_to_dynamic(%arg0: tensor<10xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %arg1) {broadcast_dimensions=array<i64: 1>}: (tensor<10xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @hlo_dynamic_broadcast_in_dim_static_to_dynamic
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<2xi32>)
//       CHECK:     %[[v0:.+]] = tensorrt.broadcast %[[arg0]] broadcast_dims<1> shape(%[[arg1]] : tensor<2xi32>)
//       CHECK:     return %[[v0]]

// -----

func.func @hlo_dynamic_broadcast_in_dim_dynamic_to_dynamic(%arg0: tensor<?x?xf32>, %arg1: tensor<3xi32>) -> tensor<?x?x?xf32> {
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %arg1) {broadcast_dimensions=array<i64: 0, 2>}: (tensor<?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// CHECK-LABEL: @hlo_dynamic_broadcast_in_dim_dynamic_to_dynamic
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?xf32>, %[[arg1:.+]]: tensor<3xi32>)
//       CHECK:     %[[v0:.+]] = tensorrt.broadcast %[[arg0]] broadcast_dims<0, 2> shape(%[[arg1]] : tensor<3xi32>)
//       CHECK:     return %[[v0]]

// -----

func.func @hlo_convert(%arg0: tensor<10xf32>) -> tensor<10xf16> {
  %0 = "stablehlo.convert"(%arg0):(tensor<10xf32>)->tensor<10xf16>
  return %0 : tensor<10xf16>
}

// CHECK-LABEL: @hlo_convert
//  CHECK-NEXT:   tensorrt.identity %{{.+}} : tensor<10xf32> to tensor<10xf16>

// -----

// -----

func.func @hlo_select(%arg0: tensor<10xi1>, %arg1: tensor<10xf32>, %arg2: tensor<10xf32>) -> tensor<10xf32> {
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2):(tensor<10xi1>, tensor<10xf32>,tensor<10xf32>)->tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @hlo_select
//  CHECK-NEXT:   tensorrt.select

// -----

func.func @hlo_select_i1(%arg0: tensor<10xi1>, %arg1: tensor<10xi1>, %arg2: tensor<10xi1>) -> tensor<10xi1> {
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2):(tensor<10xi1>, tensor<10xi1>,tensor<10xi1>)->tensor<10xi1>
  return %0 : tensor<10xi1>
}

// CHECK-LABEL: @hlo_select_i1
//  CHECK-NEXT:  tensorrt.identity
//  CHECK-SAME:  tensor<10xi1> to tensor<10xf32>
//  CHECK-NEXT:  tensorrt.identity
//  CHECK-SAME:  tensor<10xi1> to tensor<10xf32>
//  CHECK-NEXT:  tensorrt.select
//  CHECK-SAME:  ins(%{{.+}}, %{{.+}}, %{{.+}} : tensor<10xi1>, tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
//  CHECK-NEXT:  tensorrt.identity
//  CHECK-SAME:  tensor<10xf32> to tensor<10xi1>

// -----

func.func @hlo_concatenate(%arg0: tensor<10x1xf32>, %arg1: tensor<10x2xf32>, %arg2: tensor<10x3xf32>) -> tensor<10x6xf32> {
  %0 = "stablehlo.concatenate"(%arg0, %arg1, %arg2){dimension=1}:(tensor<10x1xf32>, tensor<10x2xf32>,tensor<10x3xf32>)->tensor<10x6xf32>
  return %0 : tensor<10x6xf32>
}

// CHECK-LABEL: @hlo_concatenate
//  CHECK-NEXT:   tensorrt.concatenation

// -----

func.func @hlo_transpose(%arg0: tensor<10x20x30xf32>) -> tensor<30x20x10xf32> {
  %0 = "stablehlo.transpose"(%arg0){permutation=array<i64: 2,1,0>}:(tensor<10x20x30xf32>)->tensor<30x20x10xf32>
  return %0 : tensor<30x20x10xf32>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2) -> (d2, d1, d0)>
// CHECK-LABEL: @hlo_transpose
//  CHECK-NEXT:   tensorrt.transpose
//  CHECK-SAME: {permutation = #[[$map]]}

// -----

func.func @hlo_dot_general(%arg0: tensor<?x?x32x64xf32>, %arg1: tensor<?x?x64x100xf32>) -> tensor<?x?x32x100xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [2]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<?x?x32x64xf32>, tensor<?x?x64x100xf32>) -> tensor<?x?x32x100xf32>
  return %0 : tensor<?x?x32x100xf32>
}

// CHECK-LABEL: @hlo_dot_general
//       CHECK:   tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>}


// -----

func.func @hlo_dot_general1(%arg0: tensor<?x?x64x32xf32>, %arg1: tensor<?x?x64x100xf32>) -> tensor<?x?x32x100xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [2]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<?x?x64x32xf32>, tensor<?x?x64x100xf32>) -> tensor<?x?x32x100xf32>
  return %0 : tensor<?x?x32x100xf32>
}

// CHECK-LABEL: @hlo_dot_general1
//       CHECK:   tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kTRANSPOSE>, op1 = #tensorrt.matrix_operation<kNONE>}

// -----

func.func @hlo_dot_general2(%arg0: tensor<?x?x64xf32>, %arg1: tensor<?x?x64x100xf32>) -> tensor<?x?x100xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [2]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<?x?x64xf32>, tensor<?x?x64x100xf32>) -> tensor<?x?x100xf32>
  return %0 : tensor<?x?x100xf32>
}

// CHECK-LABEL: @hlo_dot_general2
//       CHECK:   tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kVECTOR>, op1 = #tensorrt.matrix_operation<kNONE>}

// -----

func.func @main(%arg0: tensor<1x1500x384xf32>, %arg1: tensor<384x384xf32>) -> tensor<1x1500x384xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<1x1500x384xf32>, tensor<384x384xf32>) -> tensor<1x1500x384xf32>
  return %0 : tensor<1x1500x384xf32>
}
// -----

func.func @hlo_dot_general3(%arg0: tensor<32x49x32xf32>, %arg1: tensor<32x1x32x49xf32>) -> tensor<32x49x1x49xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [2]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<32x49x32xf32>, tensor<32x1x32x49xf32>) -> tensor<32x49x1x49xf32>
  return %0 : tensor<32x49x1x49xf32>
}

// CHECK-LABEL: @hlo_dot_general3
//  CHECK-NEXT:  tensorrt.expand_rank
//  CHECK-SAME:   tensor<32x49x32xf32> to tensor<32x1x49x32xf32>
//  CHECK-NEXT:  tensorrt.matrix_multiply
//  CHECK-SAME:   {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>}
//  CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<32x1x49x32xf32>, tensor<32x1x32x49xf32>) -> tensor<32x1x49x49xf32>
//  CHECK-NEXT:  tensorrt.transpose
//  CHECK-SAME:   tensor<32x1x49x49xf32> to tensor<32x49x1x49xf32>

// -----

func.func @hlo_dot_general4(%arg0: tensor<32x5x49x32xf32>, %arg1: tensor<32x5x1x32x49xf32>) -> tensor<32x5x49x1x49xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [3]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<32x5x49x32xf32>, tensor<32x5x1x32x49xf32>) -> tensor<32x5x49x1x49xf32>
  return %0 : tensor<32x5x49x1x49xf32>
}

// CHECK-LABEL: @hlo_dot_general4
//  CHECK-NEXT:  tensorrt.expand_rank
//  CHECK-SAME:   tensor<32x5x49x32xf32> to tensor<32x5x1x49x32xf32>
//  CHECK-NEXT:  tensorrt.matrix_multiply
//  CHECK-SAME:   {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>}
//  CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<32x5x1x49x32xf32>, tensor<32x5x1x32x49xf32>) -> tensor<32x5x1x49x49xf32>
//  CHECK-NEXT:  tensorrt.transpose
//  CHECK-SAME:   tensor<32x5x1x49x49xf32> to tensor<32x5x49x1x49xf32>

// -----

func.func @hlo_einsum(%arg0: tensor<?x?x128x64xf32>, %arg1: tensor<64x256xf32>) -> tensor<?x?x128x256xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "abcd,de->abce"} :
    (tensor<?x?x128x64xf32>, tensor<64x256xf32>) -> tensor<?x?x128x256xf32>
  return %0 : tensor<?x?x128x256xf32>
}

// CHECK-LABEL: @hlo_einsum
//       CHECK:   tensorrt.einsum {equation = "abcd,de->abce"}

// -----

func.func @hlo_einsum_replace_upper(%arg0: tensor<?x?x128x64xf32>, %arg1: tensor<64x256xf32>) -> tensor<?x?x128x256xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "ABcd,de->ABce"} :
    (tensor<?x?x128x64xf32>, tensor<64x256xf32>) -> tensor<?x?x128x256xf32>
  return %0 : tensor<?x?x128x256xf32>
}

// CHECK-LABEL: @hlo_einsum_replace_upper
//       CHECK:   tensorrt.einsum {equation = "abcd,de->abce"}

// -----

func.func @hlo_einsum_replace_upper(%arg0: tensor<?x?x128x64xf32>, %arg1: tensor<64x256xf32>) -> tensor<?x?x128x256xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "abcA,AB->abcB"} :
    (tensor<?x?x128x64xf32>, tensor<64x256xf32>) -> tensor<?x?x128x256xf32>
  return %0 : tensor<?x?x128x256xf32>
}

// CHECK-LABEL: @hlo_einsum_replace_upper
//       CHECK:   tensorrt.einsum {equation = "abcd,de->abce"}

// -----

func.func @hlo_einsum_ellipses_replacement(%arg0: tensor<?x?x128x64xf32>, %arg1: tensor<64x256xf32>) -> tensor<?x?x128x256xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "...cd,de->...ce"} :
    (tensor<?x?x128x64xf32>, tensor<64x256xf32>) -> tensor<?x?x128x256xf32>
  return %0 : tensor<?x?x128x256xf32>
}

// CHECK-LABEL: @hlo_einsum_ellipses_replacement
//       CHECK:   tensorrt.einsum {equation = "abcd,de->abce"}

// -----

func.func @hlo_einsum_ellipses_replacement1(%arg0: tensor<?x?x?x64xf32>, %arg1: tensor<64x256xf32>) -> tensor<?x?x?x256xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "...d,de->...e"} :
    (tensor<?x?x?x64xf32>, tensor<64x256xf32>) -> tensor<?x?x?x256xf32>
  return %0 : tensor<?x?x?x256xf32>
}

// CHECK-LABEL: @hlo_einsum_ellipses_replacement1
//       CHECK:   tensorrt.einsum {equation = "abcd,de->abce"}

// -----

func.func @hlo_einsum_ellipses_replacement2(%arg0: tensor<10x20x?x256xf32>, %arg1: tensor<10x20x64x256xf32>) -> tensor<10x20x?x64xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "xy...z,xyaz->xy...a"} :
    (tensor<10x20x?x256xf32>, tensor<10x20x64x256xf32>) -> tensor<10x20x?x64xf32>
  return %0 : tensor<10x20x?x64xf32>
}

// CHECK-LABEL: @hlo_einsum_ellipses_replacement2
//       CHECK:   tensorrt.einsum {equation = "xybz,xyaz->xyba"}

// -----

func.func @hlo_logistic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.logistic"(%arg0) {} : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: @hlo_logistic
//  CHECK-NEXT:   tensorrt.activation
//  CHECK-SAME:   activationType = #tensorrt.activation_type<kSIGMOID>
//  CHECK-NOT:   alpha
//  CHECK-NOT:   beta

// -----

func.func @hlo_sort(%arg0: tensor<1x20x10xi32>) -> (tensor<1x20x10xi32>, tensor<1x20x10xi32>) {
  %0 = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<10xi32>
  %1 = "stablehlo.broadcast_in_dim"(%0) {broadcast_dimensions = array<i64: 2>} : (tensor<10xi32>) -> tensor<1x20x10xi32>
  %2:2 = "stablehlo.sort"(%arg0, %1) ({
    ^bb0(%arg393: tensor<i32>, %arg394: tensor<i32>, %arg395: tensor<i32>, %arg396: tensor<i32>):
      %7103 = stablehlo.compare  GT, %arg393, %arg394,  NOTYPE : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %7103 : tensor<i1>
  }) {dimension = 2 : i64, is_stable = true} : (tensor<1x20x10xi32>, tensor<1x20x10xi32>) -> (tensor<1x20x10xi32>, tensor<1x20x10xi32>)
  return %2#0, %2#1 : tensor<1x20x10xi32>, tensor<1x20x10xi32>
}

// CHECK-LABEL: @hlo_sort
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x20x10xi32>)
//       CHECK:   %[[arg0_f32:.+]] = tensorrt.identity %[[arg0]] : tensor<1x20x10xi32> to tensor<1x20x10xf32>
//       CHECK:   tensorrt.top_k <kMAX>
//  CHECK-SAME:   {axis = 2 : i64, k = 10 : i64} %[[arg0_f32]] : tensor<1x20x10xf32> -> tensor<1x20x10xf32>, tensor<1x20x10xi32>
//       CHECK:   tensorrt.identity

// -----

func.func @hlo_sort1(%arg0: tensor<1x20x10xi32>) -> (tensor<1x20x10xi32>, tensor<1x20x10xi32>) {
  %0 = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<10xi32>
  %1 = "stablehlo.broadcast_in_dim"(%0) {broadcast_dimensions = array<i64: 2>} : (tensor<10xi32>) -> tensor<1x20x10xi32>
  %2:2 = "stablehlo.sort"(%arg0, %1) ({
    ^bb0(%arg393: tensor<i32>, %arg394: tensor<i32>, %arg395: tensor<i32>, %arg396: tensor<i32>):
      %7103 = stablehlo.compare  LT, %arg393, %arg394,  NOTYPE : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %7103 : tensor<i1>
  }) {dimension = 2 : i64, is_stable = true} : (tensor<1x20x10xi32>, tensor<1x20x10xi32>) -> (tensor<1x20x10xi32>, tensor<1x20x10xi32>)
  return %2#0, %2#1 : tensor<1x20x10xi32>, tensor<1x20x10xi32>
}

// CHECK-LABEL: @hlo_sort1
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x20x10xi32>)
//       CHECK:   %[[arg0_f32:.+]] = tensorrt.identity %[[arg0]] : tensor<1x20x10xi32> to tensor<1x20x10xf32>
//       CHECK:   tensorrt.top_k <kMIN>
//  CHECK-SAME:   {axis = 2 : i64, k = 10 : i64} %[[arg0_f32]] : tensor<1x20x10xf32> -> tensor<1x20x10xf32>, tensor<1x20x10xi32>

// -----

func.func @hlo_sort2() -> tensor<3xi32> {
    %cst = stablehlo.constant dense<[3, 1, 2]> : tensor<3xi32>
    %1 = "stablehlo.sort"(%cst) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %2 = stablehlo.compare  LT, %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    }) {dimension = 0 : i64, is_stable = false} : (tensor<3xi32>) -> tensor<3xi32>
    return %1 : tensor<3xi32>
}

// CHECK-LABEL: @hlo_sort2
//  CHECK-NEXT: %[[cst_i32:.+]] = tensorrt.constant
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.identity %[[cst_i32]]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.expand_rank %[[v0]]
//  CHECK-NEXT: %[[values:.+]], %[[indices:.+]] = tensorrt.top_k <kMIN> {{.*}} %[[v1]]
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.identity %[[values]]
//  CHECK-NEXT: %[[v3:.+]] = tensorrt.collapse_rank %[[v2]]
//  CHECK-NEXT: return %[[v3]]

// -----

func.func @hlo_sort3() -> tensor<2x2xi32> {
    %cst = stablehlo.constant dense<[[1,2], [3, 4]]> : tensor<2x2xi32>
    %1 = "stablehlo.sort"(%cst) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %2 = stablehlo.compare  LT, %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    }) {dimension = 0 : i64, is_stable = false} : (tensor<2x2xi32>) -> tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
}

// CHECK-LABEL: @hlo_sort3
//  CHECK-NEXT: %[[cst_i32:.+]] = tensorrt.constant
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.identity %[[cst_i32]]
//  CHECK-NEXT: %[[values:.+]], %[[indices:.+]] = tensorrt.top_k <kMIN> {{.*}} %[[v0]]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.identity %[[values]]
//  CHECK-NEXT: return %[[v1]]

// -----

func.func @hlo_sort4(%arg0: tensor<10xi32>) -> (tensor<10xi32>, tensor<10xi32>) {
  %0 = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<10xi32>
  %2:2 = "stablehlo.sort"(%arg0, %0) ({
    ^bb0(%arg393: tensor<i32>, %arg394: tensor<i32>, %arg395: tensor<i32>, %arg396: tensor<i32>):
      %7103 = stablehlo.compare  GT, %arg393, %arg394,  NOTYPE : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %7103 : tensor<i1>
  }) {dimension = 0 : i64, is_stable = true} : (tensor<10xi32>, tensor<10xi32>) -> (tensor<10xi32>, tensor<10xi32>)
  return %2#0, %2#1 : tensor<10xi32>, tensor<10xi32>
}

// CHECK-LABEL: @hlo_sort4
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}})
//       CHECK: %[[v1:.+]] = tensorrt.identity %[[arg0]]
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.expand_rank %[[v1]]
//  CHECK-NEXT: %[[values:.+]], %[[indices:.+]] = tensorrt.top_k <kMAX> {{.*}} %[[v2]]
//  CHECK-NEXT: %[[v3:.+]] = tensorrt.identity %[[values]]
//  CHECK-NEXT: %[[v4:.+]] = tensorrt.collapse_rank %[[v3]]
//  CHECK-NEXT: %[[v5:.+]] = tensorrt.collapse_rank %[[indices]]
//  CHECK-NEXT: return %[[v4]], %[[v5]]

// -----

func.func @hlo_rsqrt(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = stablehlo.rsqrt %arg0 : tensor<10xf32>
  return %0 : tensor<10xf32>
}
// CHECK-LABEL: @hlo_rsqrt
//  CHECK-NEXT:   %[[v0:.+]] = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kRECIP>} %{{.+}}
//  CHECK-NEXT:   %[[v1:.+]] = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kSQRT>} %[[v0]]
//  CHECK-NEXT:   return %[[v1]] :

// -----

func.func @hlo_reduce_sum(%arg0: tensor<1x10x20xf32>) -> tensor<1x10xf32> {
  %cst = tensorrt.constant dense<-0.000000e+00> : tensor<f32>
  %0 = stablehlo.reduce(%arg0 init: %cst)
      across dimensions = [2] : (tensor<1x10x20xf32>, tensor<f32>) -> tensor<1x10xf32>
    reducer(%arg393: tensor<f32>, %arg394: tensor<f32>)  {
      %8875 = stablehlo.add %arg393, %arg394 : tensor<f32>
      stablehlo.return %8875 : tensor<f32>
  }
  return %0 : tensor<1x10xf32>
}

// CHECK-LABEL: @hlo_reduce_sum
//  CHECK-NEXT:   tensorrt.constant
//  CHECK-NEXT:   tensorrt.reduce
//   CHECK-NOT:     keepDimensions = true
//  CHECK-SAME:     reduceAxes = array<i64: 2>
//  CHECK-SAME:     -> tensor<1x10xf32>

// -----

func.func @hlo_reduce_prod(%arg0: tensor<1x10x20xf32>) -> tensor<1x10xf32> {
  %cst = tensorrt.constant dense<-0.000000e+00> : tensor<f32>
  %0 = stablehlo.reduce(%arg0 init: %cst)
      across dimensions = [2] : (tensor<1x10x20xf32>, tensor<f32>) -> tensor<1x10xf32>
    reducer(%arg393: tensor<f32>, %arg394: tensor<f32>)  {
      %8875 = stablehlo.multiply %arg393, %arg394 : tensor<f32>
      stablehlo.return %8875 : tensor<f32>
  }
  return %0 : tensor<1x10xf32>
}

// CHECK-LABEL: @hlo_reduce_prod
//  CHECK-NEXT:   tensorrt.constant
//  CHECK-NEXT:   tensorrt.reduce
//   CHECK-NOT:     keepDimensions = true
//  CHECK-SAME:     reduceAxes = array<i64: 2>
//  CHECK-SAME:     -> tensor<1x10xf32>

// -----

func.func @hlo_reduce_or(%arg0: tensor<1x10x20xi1>) -> tensor<1x10xi1> {
  %cst = tensorrt.constant dense<0> : tensor<i32>
  %cst_i1 = tensorrt.identity %cst : tensor<i32> to tensor<i1>
  %0 = stablehlo.reduce(%arg0 init: %cst_i1)
      across dimensions = [2] : (tensor<1x10x20xi1>, tensor<i1>) -> tensor<1x10xi1>
    reducer(%arg393: tensor<i1>, %arg394: tensor<i1>)  {
      %8875 = stablehlo.or %arg393, %arg394 : tensor<i1>
      stablehlo.return %8875 : tensor<i1>
  }
  return %0 : tensor<1x10xi1>
}

// CHECK-LABEL: @hlo_reduce_or
//  CHECK-SAME:   (%[[arg0:.+]]: tensor<1x10x20xi1>)
//       CHECK:   %[[i32:.+]] = tensorrt.identity %[[arg0]] : tensor<1x10x20xi1> to tensor<1x10x20xi32>
//       CHECK:   %[[i32_sum:.+]] = tensorrt.reduce <kSUM> %[[i32]]
//  CHECK-SAME:     reduceAxes = array<i64: 2>
//  CHECK-SAME:     : tensor<1x10x20xi32> -> tensor<1x10xi32>
//       CHECK:   tensorrt.identity %[[i32_sum]] : tensor<1x10xi32> to tensor<1x10xi1>

// -----

func.func @hlo_reduce_and(%arg0: tensor<1x10x20xi1>) -> tensor<1x10xi1> {
  %cst = tensorrt.constant dense<1> : tensor<i32>
  %cst_i1 = tensorrt.identity %cst : tensor<i32> to tensor<i1>
  %0 = stablehlo.reduce(%arg0 init: %cst_i1)
      across dimensions = [2] : (tensor<1x10x20xi1>, tensor<i1>) -> tensor<1x10xi1>
    reducer(%arg393: tensor<i1>, %arg394: tensor<i1>)  {
      %8875 = stablehlo.and %arg393, %arg394 : tensor<i1>
      stablehlo.return %8875 : tensor<i1>
  }
  return %0 : tensor<1x10xi1>
}

// CHECK-LABEL: @hlo_reduce_and
//  CHECK-SAME:   (%[[arg0:.+]]: tensor<1x10x20xi1>)
//       CHECK:   %[[i32:.+]] = tensorrt.identity %[[arg0]] : tensor<1x10x20xi1> to tensor<1x10x20xi32>
//       CHECK:   %[[i32_sum:.+]] = tensorrt.reduce <kPROD> %[[i32]]
//   CHECK-NOT:     keepDimensions = true
//  CHECK-SAME:     reduceAxes = array<i64: 2>
//  CHECK-SAME:     : tensor<1x10x20xi32> -> tensor<1x10xi32>
//       CHECK:   tensorrt.identity %[[i32_sum]] : tensor<1x10xi32> to tensor<1x10xi1>


// -----

func.func @hlo_reduce_sum_multiple1(%arg0: tensor<1x10x20xf32>) -> tensor<1xf32> {
  %cst = tensorrt.constant dense<-0.000000e+00> : tensor<f32>
  %0 = stablehlo.reduce(%arg0 init: %cst)
      across dimensions = [1, 2] : (tensor<1x10x20xf32>, tensor<f32>) -> tensor<1xf32>
    reducer(%arg393: tensor<f32>, %arg394: tensor<f32>)  {
      %8875 = stablehlo.add %arg393, %arg394 : tensor<f32>
      stablehlo.return %8875 : tensor<f32>
  }
  return %0 : tensor<1xf32>
}

// CHECK-LABEL: @hlo_reduce_sum_multiple1
//  CHECK-NEXT:   tensorrt.constant
//  CHECK-NEXT:   %[[reshaped:.+]] = tensorrt.reshape %{{.+}} : tensor<1x10x20xf32> to tensor<1x200xf32>
//  CHECK-NEXT:   tensorrt.reduce <kSUM> %[[reshaped]]
//   CHECK-NOT:     keepDimensions = true
//  CHECK-SAME:     reduceAxes = array<i64: 1>
//  CHECK-SAME:      : tensor<1x200xf32> -> tensor<1xf32>

// -----

func.func @hlo_reduce_sum_multiple2(%arg0: tensor<1x112x112x64xf32>) -> (tensor<64xf32>) {
  %0 = stablehlo.constant dense<1.254400e+04> : tensor<64xf32>
  %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %2 = stablehlo.reduce(%arg0 init: %1) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x112x112x64xf32>, tensor<f32>) -> tensor<64xf32>
  %3 = stablehlo.divide %2, %0 : tensor<64xf32>
  return %3 : tensor<64xf32>
}

// CHECK-LABEL: func.func @hlo_reduce_sum_multiple2
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x112x112x64xf32>) -> tensor<64xf32>
//       CHECK:   %[[cst_f32:.+]] = tensorrt.constant dense<1.254400e+04> : tensor<64xf32>
//       CHECK:   %[[cst_f32_0:.+]] = tensorrt.constant dense<0.000000e+00> : tensor<f32>
//       CHECK:   %[[v0:.+]] = tensorrt.reshape %[[arg0]] : tensor<1x112x112x64xf32> to tensor<12544x64xf32>
//       CHECK:   %[[v1:.+]] = tensorrt.reduce <kSUM> %[[v0]] {reduceAxes = array<i64: 0>} : tensor<12544x64xf32> -> tensor<64xf32>
//       CHECK:   %[[v2:.+]] = tensorrt.element_wise <kDIV>(%[[v1]], %[[cst_f32]] : tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
//       CHECK:   return %[[v2]] : tensor<64xf32>

// -----

func.func  @hlo_reduce_sum_multiple3(%arg0: tensor<1x7x7x2048xf32>) -> (tensor<1x2048xf32>) {
  %0 = stablehlo.constant dense<4.900000e+01> : tensor<1x2048xf32>
  %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %2 = stablehlo.reduce(%arg0 init: %1) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x7x7x2048xf32>, tensor<f32>) -> tensor<1x2048xf32>
  %3 = stablehlo.divide %2, %0 : tensor<1x2048xf32>
  return %3 : tensor<1x2048xf32>
}

// CHECK-LABEL: @hlo_reduce_sum_multiple3
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x7x7x2048xf32>) -> tensor<1x2048xf32>
//       CHECK:     %[[cst_f32:.+]] = tensorrt.constant dense<4.900000e+01> : tensor<1x2048xf32>
//       CHECK:     %[[cst_f32_0:.+]] = tensorrt.constant dense<0.000000e+00> : tensor<f32>
//       CHECK:     %[[v0:.+]] = tensorrt.reshape %[[arg0]] : tensor<1x7x7x2048xf32> to tensor<1x49x2048xf32>
//       CHECK:     %[[v1:.+]] = tensorrt.reduce <kSUM> %[[v0]]
//  CHECK-SAME:        reduceAxes = array<i64: 1>
//  CHECK-SAME:         : tensor<1x49x2048xf32> -> tensor<1x2048xf32>
//       CHECK:     %[[v2:.+]] = tensorrt.element_wise <kDIV>(%[[v1]], %[[cst_f32]] :
//       CHECK:     return %[[v2]]

// -----

func.func @hlo_argmax(%arg0: tensor<1x10x20xf32>) -> (tensor<1x10xf32>, tensor<1x10xi32>) {
  %cst = tensorrt.constant dense<-0.000000e+00> : tensor<f32>
  %cst_i32 = tensorrt.constant dense<0> : tensor<i32>

  %0 = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<20xi32>
  %2 = tensorrt.broadcast %0 broadcast_dims<2> : tensor<20xi32> to tensor<1x10x20xi32>

  %4:2 = stablehlo.reduce(%arg0 init: %cst), (%2 init: %cst_i32) across dimensions = [2] :
    (tensor<1x10x20xf32>, tensor<1x10x20xi32>, tensor<f32>, tensor<i32>) -> (tensor<1x10xf32>, tensor<1x10xi32>)
     reducer(%arg393: tensor<f32>, %arg395: tensor<f32>) (%arg394: tensor<i32>, %arg396: tensor<i32>)  {
      %7212 = stablehlo.compare  GE, %arg393, %arg395,  NOTYPE : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %7213 = stablehlo.maximum %arg393, %arg395 : tensor<f32>
      %7214 = stablehlo.compare  EQ, %arg393, %arg395,  NOTYPE : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %7215 = stablehlo.minimum %arg394, %arg396 : tensor<i32>
      %7216 = stablehlo.select %7212, %arg394, %arg396 : tensor<i1>, tensor<i32>
      %7217 = stablehlo.select %7214, %7215, %7216 : tensor<i1>, tensor<i32>
      stablehlo.return %7213, %7217 : tensor<f32>, tensor<i32>
    }
  return %4#0, %4#1 : tensor<1x10xf32>, tensor<1x10xi32>
}
// CHECK-LABEL: @hlo_argmax
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<{{.+}}>)
//       CHECK:  %[[vals:.+]], %[[inds:.+]] = tensorrt.argmax {axis = 2 : i64} %[[arg0]] : tensor<1x10x20xf32> -> tensor<1x10x1xf32>, tensor<1x10x1xi32>
//       CHECK:  %[[vals_red:.+]] = tensorrt.collapse_rank %[[vals]]
//       CHECK:  %[[inds_red:.+]] = tensorrt.collapse_rank %[[inds]]
//       CHECK:  return %[[vals_red]], %[[inds_red]]

// -----

func.func @hlo_argmin(%arg0: tensor<1x10x20xf32>) -> (tensor<1x10xf32>, tensor<1x10xi32>) {
  %cst = tensorrt.constant dense<-0.000000e+00> : tensor<f32>
  %cst_i32 = tensorrt.constant dense<0> : tensor<i32>

  %0 = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<20xi32>
  %2 = tensorrt.broadcast %0 broadcast_dims<2> : tensor<20xi32> to tensor<1x10x20xi32>

  %4:2 = stablehlo.reduce(%arg0 init: %cst), (%2 init: %cst_i32) across dimensions = [2] :
    (tensor<1x10x20xf32>, tensor<1x10x20xi32>, tensor<f32>, tensor<i32>) -> (tensor<1x10xf32>, tensor<1x10xi32>)
     reducer(%arg393: tensor<f32>, %arg395: tensor<f32>) (%arg394: tensor<i32>, %arg396: tensor<i32>)  {
      %7212 = stablehlo.compare  LE, %arg393, %arg395,  NOTYPE : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %7213 = stablehlo.minimum %arg393, %arg395 : tensor<f32>
      %7214 = stablehlo.compare  EQ, %arg393, %arg395,  NOTYPE : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %7215 = stablehlo.minimum %arg394, %arg396 : tensor<i32>
      %7216 = stablehlo.select %7212, %arg394, %arg396 : tensor<i1>, tensor<i32>
      %7217 = stablehlo.select %7214, %7215, %7216 : tensor<i1>, tensor<i32>
      stablehlo.return %7213, %7217 : tensor<f32>, tensor<i32>
    }
  return %4#0, %4#1 : tensor<1x10xf32>, tensor<1x10xi32>
}

// CHECK-LABEL: @hlo_argmin
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<{{.+}}>)
//       CHECK:  %[[vals:.+]], %[[inds:.+]] = tensorrt.argmin {axis = 2 : i64} %[[arg0]] : tensor<1x10x20xf32> -> tensor<1x10x1xf32>, tensor<1x10x1xi32>
//       CHECK:  %[[vals_red:.+]] = tensorrt.collapse_rank %[[vals]]
//       CHECK:  %[[inds_red:.+]] = tensorrt.collapse_rank %[[inds]]
//       CHECK:  return %[[vals_red]], %[[inds_red]]

// -----

func.func @hlo_iota() -> tensor<128xi32> {
  %0 = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<128xi32>
  return %0 : tensor<128xi32>
}

// CHECK-LABEL: @hlo_iota
//       CHECK:  tensorrt.linspace
//  CHECK-SAME:    [ 0.00{{.+}}] [ static] [ 1.000{{.+}}] : tensor<128xi32>

// -----

func.func @hlo_rem(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) -> tensor<128xi32> {
  %0 = "stablehlo.remainder"(%arg0, %arg1) {} : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi32>
  return %0 : tensor<128xi32>
}

// CHECK-LABEL: @hlo_rem
//  CHECK-SAME:  (%[[lhs:.+]]: tensor<128xi32>, %[[rhs:.+]]: tensor<128xi32>)
//       CHECK:   %[[div:.+]] = tensorrt.element_wise <kDIV>(%[[lhs]], %[[rhs]] :
//       CHECK:   %[[prod:.+]] = tensorrt.element_wise <kPROD>(%[[div]], %[[rhs]] :
//       CHECK:   tensorrt.element_wise <kSUB>(%[[lhs]], %[[prod]] :

// -----

func.func @hlo_rem_fp(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %0 = "stablehlo.remainder"(%arg0, %arg1) {} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
  return %0 : tensor<128xf32>
}

// CHECK-LABEL: @hlo_rem_fp
//  CHECK-SAME:  (%[[lhs:.+]]: tensor<128xf32>, %[[rhs:.+]]: tensor<128xf32>)
//       CHECK:   %[[div:.+]] = tensorrt.element_wise <kDIV>(%[[lhs]], %[[rhs]] :
//       CHECK:   %[[i32:.+]] = tensorrt.identity %[[div]] : tensor<128xf32> to tensor<128xi32>
//       CHECK:   %[[f32:.+]] = tensorrt.identity %[[i32]] : tensor<128xi32> to tensor<128xf32>
//       CHECK:   %[[prod:.+]] = tensorrt.element_wise <kPROD>(%[[f32]], %[[rhs]] :
//       CHECK:   tensorrt.element_wise <kSUB>(%[[lhs]], %[[prod]] :

// -----

func.func @torch_index_select(%arg0: tensor<5x1x5xi32>,
                              %arg1: tensor<2xi32>) ->  tensor<2x1x5xi32> {
  %0 = "stablehlo.torch_index_select"(%arg0, %arg1) {
    dim = 0 : i64,
    batch_dims = 0 : i64
  } : (tensor<5x1x5xi32>, tensor<2xi32>) -> tensor<2x1x5xi32>
  func.return %0 : tensor<2x1x5xi32>
}

// CHECK-LABEL: @torch_index_select
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<{{.+}}>, %[[arg1:.+]]: tensor
//       CHECK:  tensorrt.gather {axis = 0 : i64} ins(%[[arg0]], %[[arg1]]

// -----

// TODO: support this case with simple rank expansion.
func.func @torch_index_select_scalar(%arg0: tensor<4x8xf32>,
                                %arg1: tensor<i32>) -> tensor<8xf32> {
  %0 = "stablehlo.torch_index_select"(%arg0, %arg1) {
    batch_dims = 0 : i64,
    dim = 0 : i64
  } : (tensor<4x8xf32>, tensor<i32>) -> tensor<8xf32>
  func.return %0 : tensor<8xf32>
}

// CHECK-LABEL: @torch_index_select_scalar
//   CHECK-NOT:   tensorrt.gather
//       CHECK:   stablehlo.torch_index_select

// -----

func.func @torch_index_select_batch(%arg0: tensor<4x7x8x2xf32>,
                               %arg1: tensor<4x1xi32>) -> tensor<4x7x1x2xf32> {
  %0 = "stablehlo.torch_index_select"(%arg0, %arg1) {
    dim = 2 : i64,
    batch_dims = 1 : i64
  } : (tensor<4x7x8x2xf32>, tensor<4x1xi32>) -> tensor<4x7x1x2xf32>
  func.return %0 : tensor<4x7x1x2xf32>
}

// CHECK-LABEL: @torch_index_select_batch
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<{{.+}}>, %[[arg1:.+]]: tensor
//       CHECK:  tensorrt.gather {axis = 2 : i64, numBroadcastDims = 1 : i64} ins(%[[arg0]], %[[arg1]]

// -----

func.func @torch_index_select_batch2(%arg0: tensor<1x10x7x8x2xf32>,
                                     %arg1: tensor<1x10x1xi32>) -> tensor<1x10x7x1x2xf32> {
  %0 = "stablehlo.torch_index_select"(%arg0, %arg1) {
    dim = 3 : i64,
    batch_dims = 2 : i64
  } : (tensor<1x10x7x8x2xf32>, tensor<1x10x1xi32>) -> tensor<1x10x7x1x2xf32>
  func.return %0 : tensor<1x10x7x1x2xf32>
}

// CHECK-LABEL: @torch_index_select_batch2
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<{{.+}}>, %[[arg1:.+]]: tensor
//       CHECK:  %[[data:.+]] = tensorrt.collapse_rank %[[arg0]]
//       CHECK:  %[[ind:.+]] = tensorrt.collapse_rank %[[arg1]]
//       CHECK:  %[[g:.+]] = tensorrt.gather {axis = 2 : i64, numBroadcastDims = 1 : i64} ins(%[[data]], %[[ind]]
//       CHECK:  %[[res:.+]] = tensorrt.expand_rank %[[g]]
//       CHECK:  return %[[res]]
// -----

func.func @torch_index_select_dynamic(%input: tensor<?x?x?x?xf32>,
                                      %index: tensor<?x?xi32>) -> tensor<?x?x?x?xf32>{
  %0 = "stablehlo.torch_index_select"(%input, %index) {
    batch_dims = 1 : i64,
    dim = 2 : i64
  } : (tensor<?x?x?x?xf32>, tensor<?x?xi32>) -> tensor<?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: @torch_index_select_dynamic
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<{{.+}}>, %[[arg1:.+]]: tensor
//       CHECK:  tensorrt.gather {axis = 2 : i64, numBroadcastDims = 1 : i64} ins(%[[arg0]], %[[arg1]]

// -----

func.func @hlo_reverse(%input: tensor<20x30xf32>) -> tensor<20x30xf32> {
  %result = "stablehlo.reverse"(%input) {
    dimensions = array<i64: 1>
  } : (tensor<20x30xf32>) -> tensor<20x30xf32>
  func.return %result : tensor<20x30xf32>
}

// CHECK-LABEL: @hlo_reverse(
//  CHECK-SAME:   %[[arg0:.+]]: tensor<20x30xf32>)
//       CHECK:   tensorrt.slice %[[arg0]][0, 29][20, 30][1, -1] : tensor<20x30xf32> to tensor<20x30xf32>

// -----

func.func @hlo_reverse(%input: tensor<20x30xf32>) -> tensor<20x30xf32> {
  %result = "stablehlo.reverse"(%input) {
    dimensions = array<i64: 0, 1>
  } : (tensor<20x30xf32>) -> tensor<20x30xf32>
  func.return %result : tensor<20x30xf32>
}

// CHECK-LABEL: @hlo_reverse(
//  CHECK-SAME:   %[[arg0:.+]]: tensor<20x30xf32>)
//       CHECK:   tensorrt.slice %[[arg0]][19, 29][20, 30][-1, -1] : tensor<20x30xf32> to tensor<20x30xf32>

// -----

func.func @hlo_pad_static(%arg0: tensor<10x48x48x32xf32>) -> tensor<10x48x48x48xf32> {
  %0 = "stablehlo.constant"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  %1 = "stablehlo.pad"(%arg0, %0) {
    edge_padding_high = array<i64: 0, 0, 0, 16>,
    edge_padding_low = array<i64: 0, 0, 0, 0>,
    interior_padding = array<i64: 0, 0, 0, 0>
  } : (tensor<10x48x48x32xf32>, tensor<f32>) -> tensor<10x48x48x48xf32>
  func.return %1 : tensor<10x48x48x48xf32>
}

// CHECK-LABEL: @hlo_pad_static
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<10x48x48x32xf32>
//       CHECK:  %[[fill:.+]] = tensorrt.constant dense<0.0{{.*}}> : tensor<f32>
//       CHECK:  tensorrt.slice %[[arg0]][0, 0, 0, 0][10, 48, 48, 48][1, 1, 1, 1] fill(%[[fill]] : tensor<f32>) {mode = #tensorrt.slice_mode<kFILL>} : tensor<10x48x48x32xf32> to tensor<10x48x48x48xf32>

// -----

func.func @hlo_pad_static_low_high(%arg0: tensor<10x48x48x32xf32>) -> tensor<10x48x48x64xf32> {
  %0 = "stablehlo.constant"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  %1 = "stablehlo.pad"(%arg0, %0) {
    edge_padding_high = array<i64: 0, 0, 0, 16>,
    edge_padding_low = array<i64: 0, 0, 0, 16>,
    interior_padding = array<i64: 0, 0, 0, 0>
  } : (tensor<10x48x48x32xf32>, tensor<f32>) -> tensor<10x48x48x64xf32>
  func.return %1 : tensor<10x48x48x64xf32>
}

// CHECK-LABEL: @hlo_pad_static
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<10x48x48x32xf32>
//       CHECK:  %[[fill:.+]] = tensorrt.constant dense<0.0{{.*}}> : tensor<f32>
//       CHECK:  tensorrt.slice %[[arg0]][0, 0, 0, -16][10, 48, 48, 64][1, 1, 1, 1] fill(%[[fill]] : tensor<f32>) {mode = #tensorrt.slice_mode<kFILL>} : tensor<10x48x48x32xf32> to tensor<10x48x48x64xf32>

// -----

func.func @hlo_pad_dynamic_non_sliced_dim(%arg0: tensor<?x48x48x32xf32>) -> tensor<?x48x48x48xf32> {
  %0 = "stablehlo.constant"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  %1 = "stablehlo.pad"(%arg0, %0) {
    edge_padding_high = array<i64: 0, 0, 0, 16>,
    edge_padding_low = array<i64: 0, 0, 0, 0>,
    interior_padding = array<i64: 0, 0, 0, 0>
  } : (tensor<?x48x48x32xf32>, tensor<f32>) -> tensor<?x48x48x48xf32>
  func.return %1 : tensor<?x48x48x48xf32>
}

// CHECK-LABEL: @hlo_pad_dynamic_non_sliced_dim
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<?x48x48x32xf32>
//       CHECK:   %[[fill:.+]] = tensorrt.constant dense<0.000000e+00> : tensor<f32>
//       CHECK:   %[[shape:.+]] = tensorrt.shape %[[arg0]] : tensor<?x48x48x32xf32> -> tensor<4xi32>
//       CHECK:   %[[pad_high:.+]] = tensorrt.constant dense<[0, 0, 0, 16]> : tensor<4xi32>
//       CHECK:   %[[pad_low:.+]] = tensorrt.constant dense<0> : tensor<4xi32>
//       CHECK:   %[[sum0:.+]] = tensorrt.element_wise <kSUM>(%[[pad_low]], %[[pad_high]] : tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
//       CHECK:   %[[sum1:.+]] = tensorrt.element_wise <kSUM>(%[[sum0]], %[[shape]] : tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
//       CHECK:   tensorrt.slice %arg0[0, 0, 0, 0][%[[sum1]]: tensor<4xi32>][1, 1, 1, 1] fill(%[[fill]] : tensor<f32>) {mode = #tensorrt.slice_mode<kFILL>} : tensor<?x48x48x32xf32> to tensor<?x48x48x48xf32>

// -----

func.func @hlo_pad_interior_unsupported(%arg0: tensor<1x2x3xf16>, %arg1: tensor<f16>) -> tensor<2x4x7xf16> {
  %0 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_high = array<i64: 1, 1, 0>,
    edge_padding_low = array<i64: 0, 1, 2>,
    interior_padding = array<i64: 0, 0, 1>
  } : (tensor<1x2x3xf16>, tensor<f16>) -> tensor<2x4x7xf16>
  func.return %0 : tensor<2x4x7xf16>
}

// CHECK-LABEL: @hlo_pad_interior_unsupported
//   CHECK-NOT:  tensorrt
//       CHECK:  stablehlo.pad

// -----

func.func @hlo_compare_ne(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10xi1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction NE>} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
  return %0 : tensor<10xi1>
}

// CHECK-LABEL: @hlo_compare_ne
//  CHECK-SAME:   (%[[arg0:.+]]: tensor<{{.*}}>, %[[arg1:.+]]: tensor<{{.*}}>)
//       CHECK:  %[[eq:.+]] = tensorrt.element_wise <kEQUAL>(%[[arg0]], %[[arg1]] :
//       CHECK:  tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kNOT>} %[[eq]]

// -----


func.func @hlo_compare_lt(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10xi1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
  return %0 : tensor<10xi1>
}

// CHECK-LABEL: @hlo_compare_lt
//  CHECK-SAME:   (%[[arg0:.+]]: tensor<{{.*}}>, %[[arg1:.+]]: tensor<{{.*}}>)
//       CHECK:  %[[eq:.+]] = tensorrt.element_wise <kLESS>(%[[arg0]], %[[arg1]] :

// -----


func.func @hlo_compare_gt(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10xi1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
  return %0 : tensor<10xi1>
}

// CHECK-LABEL: @hlo_compare_gt
//  CHECK-SAME:   (%[[arg0:.+]]: tensor<{{.*}}>, %[[arg1:.+]]: tensor<{{.*}}>)
//       CHECK:  %[[eq:.+]] = tensorrt.element_wise <kGREATER>(%[[arg0]], %[[arg1]] :

// -----


func.func @hlo_compare_le(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10xi1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction LE>} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
  return %0 : tensor<10xi1>
}

// CHECK-LABEL: @hlo_compare_le
//  CHECK-SAME:   (%[[arg0:.+]]: tensor<{{.*}}>, %[[arg1:.+]]: tensor<{{.*}}>)
//   CHECK-DAG:   %[[eq:.+]] = tensorrt.element_wise <kEQUAL>(%[[arg0]], %[[arg1]] :
//   CHECK-DAG:   %[[lt:.+]] = tensorrt.element_wise <kLESS>(%[[arg0]], %[[arg1]] :
//       CHECK:   tensorrt.element_wise <kOR>(%[[eq]], %[[lt]] :

// -----


func.func @hlo_compare_ge(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10xi1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction GE>} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
  return %0 : tensor<10xi1>
}

// CHECK-LABEL: @hlo_compare_ge
//  CHECK-SAME:   (%[[arg0:.+]]: tensor<{{.*}}>, %[[arg1:.+]]: tensor<{{.*}}>)
//   CHECK-DAG:   %[[eq:.+]] = tensorrt.element_wise <kEQUAL>(%[[arg0]], %[[arg1]] :
//   CHECK-DAG:   %[[lt:.+]] = tensorrt.element_wise <kGREATER>(%[[arg0]], %[[arg1]] :
//       CHECK:   tensorrt.element_wise <kOR>(%[[eq]], %[[lt]] :

// -----

func.func @hlo_average_pool(%arg0: tensor<1x10x200x300xf32>) -> tensor<1x10x20x10xf32>{
  %cst_0 = stablehlo.constant dense<0.0> : tensor<f32>
  %1 = "stablehlo.reduce_window"(%arg0, %cst_0) ({
    ^bb0(%arg393: tensor<f32>, %arg394: tensor<f32>):
      %8858 = stablehlo.add %arg393, %arg394 : tensor<f32>
      stablehlo.return %8858 : tensor<f32>
  }) {window_dimensions = array<i64: 1, 1, 10, 30>,
  window_strides = array<i64: 1, 1, 10, 30>}
    : (tensor<1x10x200x300xf32>, tensor<f32>) -> tensor<1x10x20x10xf32>
  return %1 : tensor<1x10x20x10xf32>
}

// CHECK-LABEL: @hlo_average_pool
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<1x10x200x300xf32>) -> tensor<1x10x20x10xf32> {
//       CHECK: %[[pool:.+]] = tensorrt.pooling {averageCountExcludesPadding = true,
//  CHECK-SAME:   poolingType = #tensorrt.pooling_type<kAVERAGE>,
//  CHECK-SAME:   postPadding = array<i64: 0, 0>, prePadding = array<i64: 0, 0>
//  CHECK-SAME:   stride = array<i64: 10, 30>, windowSize = array<i64: 10, 30>
//  CHECK-SAME:   ins(%[[arg0]] : tensor<1x10x200x300xf32>) -> tensor<1x10x20x10xf32>
//       CHECK: %[[windowVol:.+]] = tensorrt.constant dense<3.0{{.*}}>
//       CHECK: %[[prod:.+]] = tensorrt.element_wise <kPROD>(%[[pool]], %[[windowVol]]
//       CHECK: return %[[prod]] : tensor<1x10x20x10xf32>

// -----

func.func @hlo_max_pool(%arg0: tensor<1x10x200x300xf32>) -> tensor<1x10x20x10xf32>{
  %cst_0 = stablehlo.constant dense<0.0> : tensor<f32>
  %1 = "stablehlo.reduce_window"(%arg0, %cst_0) ({
    ^bb0(%arg393: tensor<f32>, %arg394: tensor<f32>):
      %8858 = stablehlo.maximum %arg393, %arg394 : tensor<f32>
      stablehlo.return %8858 : tensor<f32>
  }) {window_dimensions = array<i64: 1, 1, 10, 30>,
  window_strides = array<i64: 1, 1, 10, 30>}
    : (tensor<1x10x200x300xf32>, tensor<f32>) -> tensor<1x10x20x10xf32>
  return %1 : tensor<1x10x20x10xf32>
}

// CHECK-LABEL: @hlo_max_pool
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<1x10x200x300xf32>) -> tensor<1x10x20x10xf32> {
//       CHECK: %[[pool:.+]] = tensorrt.pooling {
//  CHECK-SAME:   poolingType = #tensorrt.pooling_type<kMAX>,
//  CHECK-SAME:   postPadding = array<i64: 0, 0>, prePadding = array<i64: 0, 0>
//  CHECK-SAME:   stride = array<i64: 10, 30>, windowSize = array<i64: 10, 30>
//  CHECK-SAME:   ins(%[[arg0]] : tensor<1x10x200x300xf32>) -> tensor<1x10x20x10xf32>
//       CHECK: return %[[pool]] : tensor<1x10x20x10xf32>

// -----

func.func  @hlo_reduce_window_requires_transpose(%arg0: tensor<1x112x112x64xf32>) -> tensor<1x56x56x64xf32> {
  %0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %1 = "stablehlo.reduce_window"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %2 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
    stablehlo.return %2 : tensor<f32>
  }) {base_dilations = array<i64: 1, 1, 1, 1>,
    padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>,
    window_dilations = array<i64: 1, 1, 1, 1>,
    window_dimensions = array<i64: 1, 3, 3, 1>,
    window_strides = array<i64: 1, 2, 2, 1>} : (tensor<1x112x112x64xf32>, tensor<f32>) -> tensor<1x56x56x64xf32>
  return %1 : tensor<1x56x56x64xf32>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
//       CHECK: #[[$map1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @hlo_reduce_window_requires_transpose
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x112x112x64xf32>) -> tensor<1x56x56x64xf32>
//       CHECK:     %[[cst_f32:.+]] = tensorrt.constant dense<0xFF800000> : tensor<f32>
//       CHECK:     %[[v0:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[arg0]] :
//       CHECK:     %[[v1:.+]] = tensorrt.pooling
//  CHECK-SAME:       poolingType = #tensorrt.pooling_type<kMAX>
//  CHECK-SAME:       postPadding = array<i64: 1, 1>
//  CHECK-SAME:       prePadding = array<i64: 0, 0>
//  CHECK-SAME:       stride = array<i64: 2, 2>
//  CHECK-SAME:       windowSize = array<i64: 3, 3>}
//  CHECK-SAME:        ins(%[[v0]] : tensor<1x64x112x112xf32>)
//  CHECK-SAME:        tensor<1x64x56x56xf32>
//       CHECK:     %[[v2:.+]] = tensorrt.transpose {permutation = #[[$map1]]} %[[v1]] : tensor<1x64x56x56xf32> to tensor<1x56x56x64xf32>
//       CHECK:     return %[[v2]] : tensor<1x56x56x64xf32>

// -----

func.func  @hlo_reduce_window_5d_requires_transpose(%arg0: tensor<1x112x112x112x64xf32>) -> tensor<1x56x56x56x64xf32> {
  %0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %1 = "stablehlo.reduce_window"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %2 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
    stablehlo.return %2 : tensor<f32>
  }) {
    base_dilations = array<i64: 1,1,1,1,1>,
    padding = dense<[[0, 0], [0, 1], [0, 1], [0, 1], [0, 0]]> : tensor<5x2xi64>,
    window_dilations = array<i64: 1, 1, 1, 1, 1>,
    window_dimensions = array<i64: 1, 3, 3, 3, 1>,
    window_strides = array<i64: 1, 2, 2, 2, 1>
  } : (tensor<1x112x112x112x64xf32>, tensor<f32>) -> tensor<1x56x56x56x64xf32>
  return %1 : tensor<1x56x56x56x64xf32>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2, d3)>
//       CHECK: #[[$map1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4, d1)>
// CHECK-LABEL: @hlo_reduce_window_5d_requires_transpose
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x112x112x112x64xf32>) -> tensor<1x56x56x56x64xf32>
//       CHECK:     %[[cst_f32:.+]] = tensorrt.constant dense<0xFF800000> : tensor<f32>
//       CHECK:     %[[v0:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[arg0]] :
//       CHECK:     %[[v1:.+]] = tensorrt.pooling
//  CHECK-SAME:       poolingType = #tensorrt.pooling_type<kMAX>
//  CHECK-SAME:       postPadding = array<i64: 1, 1, 1>
//  CHECK-SAME:       prePadding = array<i64: 0, 0, 0>
//  CHECK-SAME:       stride = array<i64: 2, 2, 2>
//  CHECK-SAME:       windowSize = array<i64: 3, 3, 3>}
//  CHECK-SAME:        ins(%[[v0]] : tensor<1x64x112x112x112xf32>)
//  CHECK-SAME:        tensor<1x64x56x56x56xf32>
//       CHECK:     %[[v2:.+]] = tensorrt.transpose {permutation = #[[$map1]]} %[[v1]] : tensor<1x64x56x56x56xf32> to tensor<1x56x56x56x64xf32>
//       CHECK:     return %[[v2]] : tensor<1x56x56x56x64xf32>

// -----

func.func @hlo_tanh(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.tanh"(%arg0) {} : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: @hlo_tanh
//  CHECK-NEXT:   tensorrt.activation
//  CHECK-SAME:   activationType = #tensorrt.activation_type<kTANH>

// -----

func.func @hlo_sign(%arg0: tensor<1x128x64xf32>) -> tensor<1x128x64xf32> {
  %0 = "stablehlo.sign"(%arg0) : (tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
  return %0 : tensor<1x128x64xf32>
}

// CHECK-LABEL: @hlo_sign
//       CHECK:   tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kSIGN>}

// -----

func.func @hlo_clamp(%lb : tensor<4xf32>, %x : tensor<4xf32>, %ub : tensor<4xf32>) -> tensor<4xf32> {
  %0 = "stablehlo.clamp"(%lb, %x, %ub) : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// CHECK-LABEL: @hlo_clamp
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xf32>, %[[arg1:.+]]: tensor<4xf32>, %[[arg2:.+]]: tensor<4xf32>) -> tensor<4xf32> {
//       CHECK: tensorrt.element_wise
//  CHECK-SAME:      <kMAX>
//       CHECK: tensorrt.element_wise
//  CHECK-SAME:      <kMIN>

// -----

func.func @hlo_scalar_clamp(%lb : tensor<f32>, %x : tensor<4xf32>, %ub : tensor<f32>) -> tensor<4xf32> {
  %0 = "stablehlo.clamp"(%lb, %x, %ub) : (tensor<f32>, tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// CHECK-LABEL: @hlo_scalar_clamp
//       CHECK: tensorrt.expand_rank
//       CHECK: tensorrt.element_wise
//  CHECK-SAME:      <kMAX>
//       CHECK: tensorrt.expand_rank
//       CHECK: tensorrt.element_wise
//  CHECK-SAME:      <kMIN>

// -----

func.func @hlo_clamp_i32(%lb : tensor<4xi32>, %x : tensor<4xi32>, %ub : tensor<4xi32>) -> tensor<4xi32> {
  %0 = "stablehlo.clamp"(%lb, %x, %ub) : (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: @hlo_clamp
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xi32>, %[[arg1:.+]]: tensor<4xi32>, %[[arg2:.+]]: tensor<4xi32>) -> tensor<4xi32> {
//       CHECK:   %[[input:.+]] = tensorrt.identity %[[arg1]] : tensor<4xi32> to tensor<4xf32>
//       CHECK:   %[[lower:.+]] = tensorrt.identity %[[arg0]] : tensor<4xi32> to tensor<4xf32>
//       CHECK:   %[[upper:.+]] = tensorrt.identity %[[arg2]] : tensor<4xi32> to tensor<4xf32>
//       CHECK:   %[[result1:.+]] = tensorrt.element_wise <kMAX>(%[[input]], %[[lower]] : tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
//       CHECK:   %[[result2:.+]] = tensorrt.element_wise <kMIN>(%[[result1]], %[[upper]] : tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
//       CHECK:   %[[result:.+]] = tensorrt.identity %[[result2]] : tensor<4xf32> to tensor<4xi32>
//       CHECK:   return %[[result]]

// -----

func.func @hlo_dynamic_pad(%arg0: tensor<?x?xf32>, %arg1: tensor<f32>, %arg2: tensor<2xi32>, %arg3: tensor<2xi32>) -> tensor<?x?xf32> {
  %cst_0 = stablehlo.constant dense<[0, 0]> : tensor<2xi32>
  %0 = "stablehlo.dynamic_pad"(%arg0, %arg1, %arg2, %arg3, %cst_0) : (tensor<?x?xf32>, tensor<f32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %0: tensor<?x?xf32>
}

// CHECK-LABEL: @hlo_dynamic_pad
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<?x?xf32>, %[[arg1:.+]]: tensor<f32>, %[[arg2:.+]]: tensor<2xi32>, %[[arg3:.+]]: tensor<2xi32>) -> tensor<?x?xf32> {
//       CHECK:   %[[pad_interior:.+]] = tensorrt.constant dense<0> : tensor<2xi32>
//       CHECK:   %[[padding_low_f32:.+]] = tensorrt.identity %arg2 : tensor<2xi32> to tensor<2xf32>
//       CHECK:   %[[sliceOffset_f32:.+]] = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kNEG>} %[[padding_low_f32]] : tensor<2xf32>
//       CHECK:   %[[sliceOffset_i32:.+]] = tensorrt.identity %[[sliceOffset_f32]] : tensor<2xf32> to tensor<2xi32>
//       CHECK:   %[[shape:.+]] = tensorrt.shape %[[arg0]] : tensor<?x?xf32> -> tensor<2xi32>
//       CHECK:   %[[sum0:.+]] = tensorrt.element_wise <kSUM>(%[[arg2]], %[[arg3]] : tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
//       CHECK:   %[[sum1:.+]] = tensorrt.element_wise <kSUM>(%[[sum0]], %[[shape]] : tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
//       CHECK:   tensorrt.slice %arg0[%[[sliceOffset_i32]]: tensor<2xi32>][%[[sum1]]: tensor<2xi32>][1, 1] fill(%arg1 : tensor<f32>) {mode = #tensorrt.slice_mode<kFILL>} : tensor<?x?xf32> to tensor<?x?xf32>

// -----

func.func @hlo_dynamic_pad_interior_unsupported(%arg0: tensor<?x?xf32>, %arg1: tensor<f32>, %arg2: tensor<2xi32>, %arg3: tensor<2xi32>, %arg4: tensor<2xi32>) -> tensor<?x?xf32> {
  %0 = "stablehlo.dynamic_pad"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<?x?xf32>, tensor<f32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %0: tensor<?x?xf32>
}

// CHECK-LABEL: @hlo_dynamic_pad_interior_unsupported
//   CHECK-NOT:  tensorrt
//       CHECK:  stablehlo.dynamic_pad

// -----

func.func @hlo_dynamic_reshape(%arg0: tensor<?xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %0 = "stablehlo.dynamic_reshape"(%arg0, %arg1) : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @hlo_dynamic_reshape
//  CHECK-SAME:    (%[[arg0:.+]]: tensor<{{.+}}>, %[[arg1:.+]]: tensor<{{.+}}>)
//       CHECK:  tensorrt.reshape %[[arg0]] shape(%[[arg1]]: tensor<2xi32>) : tensor<?xf32> to tensor<?x?xf32>

// -----

func.func @hlo_get_dimension_size(%arg0: tensor<?x?xf32>) -> tensor<i32> {
  %0 = "stablehlo.get_dimension_size"(%arg0) {
    dimension = 0 : i64
  } : (tensor<?x?xf32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// CHECK-LABEL: @hlo_get_dimension_size
//       CHECK: %[[shape:.+]] = tensorrt.shape %arg0 : tensor<?x?xf32> -> tensor<2xi32>
//       CHECK: %[[slice:.+]] = tensorrt.slice %[[shape]][0][1][1] : tensor<2xi32> to tensor<1xi32>
//       CHECK:  tensorrt.reshape %[[slice]] : tensor<1xi32> to tensor<i32>

// -----

func.func @hlo_round_nearest_even(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = "stablehlo.round_nearest_even"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: @hlo_round_nearest_even
//       CHECK:   tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kROUND>}

// -----

func.func @hlo_dynamic_iota(%arg0 : tensor<1xi32>) -> tensor<?xi32> {
  %0 = "stablehlo.dynamic_iota"(%arg0) {iota_dimension = 0 : i64} : (tensor<1xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK-LABEL: @hlo_dynamic_iota
//       CHECK:  tensorrt.linspace
//  CHECK-SAME:    [ 0.00{{.+}}] [%arg0 : tensor<1xi32>] [ 1.000{{.+}}] : tensor<?xi32>

// -----

func.func @stablehlo_broadcast(%arg0: tensor<8xf32>) -> tensor<4x8xf32> {
  %0 = "stablehlo.broadcast"(%arg0) {
    broadcast_sizes = array<i64: 4>
  } : (tensor<8xf32>) -> tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: @stablehlo_broadcast
//  CHECK-SAME:  (%arg0: tensor<8xf32>)
//       CHECK:  %[[v0:.+]] = tensorrt.expand_rank %[[arg0]] : tensor<8xf32> to tensor<1x8xf32>
//       CHECK:  %[[v1:.+]] = tensorrt.broadcast %[[v0]] broadcast_dims<0, 1> : tensor<1x8xf32> to tensor<4x8xf32>

// -----

func.func @stablehlo_real_dynamic_slice(
    %input: tensor<?x?xf32>,
    %start_indices: tensor<2xindex>,
    %limit_indices: tensor<2xindex>,
    %strides: tensor<2xindex>) -> tensor<?x?xf32> {
  %0 = "stablehlo.real_dynamic_slice"(%input, %start_indices, %limit_indices, %strides) : (tensor<?x?xf32>, tensor<2xindex>, tensor<2xindex>, tensor<2xindex>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @stablehlo_real_dynamic_slice(
//  CHECK-SAME:     %[[arg0:.+]]: tensor<?x?xf32>, %[[arg1:.+]]: tensor<2xi32>, %[[arg2:.+]]: tensor<2xi32>, %[[arg3]]: tensor<2xi32>) -> tensor<?x?xf32> {
//       CHECK:   %[[num:.+]] = tensorrt.element_wise <kSUB>(%[[arg2]], %[[arg1]]
//       CHECK:   %[[cst_i32:.+]] = tensorrt.constant dense<1> : tensor<2xi32>
//       CHECK:   %[[bias:.+]] = tensorrt.element_wise <kSUB>(%[[arg3]], %[[cst_i32]] : tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
//       CHECK:   %[[num1:.+]] = tensorrt.element_wise <kSUM>(%[[num]], %[[bias]] : tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
//       CHECK:   %[[ceilDiv:.+]] = tensorrt.element_wise <kDIV>(%[[num1]], %[[arg3]] : tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
//       CHECK:   %[[result:.+]] = tensorrt.slice %[[arg0]][%[[arg1]]: tensor<2xi32>][%[[ceilDiv]]: tensor<2xi32>][%[[arg3]]: tensor<2xi32>]
//       CHECK:   return %[[result]] : tensor<?x?xf32>

// -----

func.func @hlo_log1p(%arg0: tensor<10x20x30xf32>) -> tensor<10x20x30xf32> {
    %0 = "stablehlo.log_plus_one"(%arg0) : (tensor<10x20x30xf32>) -> tensor<10x20x30xf32>
    return %0 : tensor<10x20x30xf32>
}

// CHECK-LABEL: @hlo_log1p
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10x20x30xf32>) -> tensor<10x20x30xf32>
//       CHECK: %[[cst_f32:.+]] = tensorrt.constant dense<1.000000e+00> : tensor<1x1x1xf32>
//       CHECK: %[[u:.+]] = tensorrt.element_wise <kSUM>(%arg0, %cst_f32 : tensor<10x20x30xf32>, tensor<1x1x1xf32>) -> tensor<10x20x30xf32>
//       CHECK: %[[u_equal_const_one:.+]] = tensorrt.element_wise <kEQUAL>(%0, %cst_f32 : tensor<10x20x30xf32>, tensor<1x1x1xf32>) -> tensor<10x20x30xi1>
//       CHECK: %[[log_u:.+]] = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kLOG>} %[[u]] : tensor<10x20x30xf32>
//       CHECK: %[[u_equal_inf:.+]] = tensorrt.element_wise <kEQUAL>(%[[u]], %[[log_u]] : tensor<10x20x30xf32>, tensor<10x20x30xf32>) -> tensor<10x20x30xi1>
//       CHECK: %[[condition:.+]] = tensorrt.element_wise <kOR>(%[[u_equal_const_one]], %[[u_equal_inf]] : tensor<10x20x30xi1>, tensor<10x20x30xi1>) -> tensor<10x20x30xi1>
//       CHECK: %[[u_sub_const_one:.+]] = tensorrt.element_wise <kSUB>(%[[u]], %[[cst_f32]] : tensor<10x20x30xf32>, tensor<1x1x1xf32>) -> tensor<10x20x30xf32>
//       CHECK: %[[log_u_div_u_sub_const_one:.+]] = tensorrt.element_wise <kDIV>(%[[log_u]], %[[u_sub_const_one]] : tensor<10x20x30xf32>, tensor<10x20x30xf32>) -> tensor<10x20x30xf32>
//       CHECK: %[[large_log:.+]] = tensorrt.element_wise <kPROD>(%[[arg0]], %[[log_u_div_u_sub_const_one]] : tensor<10x20x30xf32>, tensor<10x20x30xf32>) -> tensor<10x20x30xf32>
//       CHECK: %[[approximation:.+]] = tensorrt.select ins(%[[condition]], %[[arg0]], %[[large_log]] : tensor<10x20x30xi1>, tensor<10x20x30xf32>, tensor<10x20x30xf32>) -> tensor<10x20x30xf32>

// -----

func.func @hlo_atan2(%lhs: tensor<128x128xf32>, %rhs: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = "stablehlo.atan2"(%lhs, %rhs) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-LABEL: @hlo_atan2
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<{{.+}}>, %[[arg1:.+]]: tensor<{{.+}}>) -> tensor<128x128xf32>
//       CHECK:  %[[intermediate_div:.+]] = tensorrt.element_wise <kDIV>(%[[arg0]], %[[arg1]] : tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
//       CHECK:  %[[intermediate_atan:.+]] = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kATAN>} %[[intermediate_div]] : tensor<128x128xf32>
//       CHECK:  %[[cst_f32:.+]] = tensorrt.constant dense<0.000000e+00> : tensor<1x1xf32>
//       CHECK:  %[[cst_f32_0:.+]] = tensorrt.constant dense<1.000000e+00> : tensor<1x1xf32>
//       CHECK:  %[[cst_f32_1:.+]] = tensorrt.constant dense<2.000000e+00> : tensor<1x1xf32>
//       CHECK:  %[[cst_f32_2:.+]] = tensorrt.constant dense<3.14159274> : tensor<1x1xf32>
//       CHECK:  %[[lhs_mask:.+]] = tensorrt.element_wise <kLESS>(%[[arg0]], %[[cst_f32]] : tensor<128x128xf32>, tensor<1x1xf32>) -> tensor<128x128xi1>
//       CHECK:  %[[rhs_mask:.+]] = tensorrt.element_wise <kLESS>(%[[arg1]], %[[cst_f32]] : tensor<128x128xf32>, tensor<1x1xf32>) -> tensor<128x128xi1>
//       CHECK:  %[[lhs_mask_fp32:.+]] = tensorrt.identity %[[lhs_mask]] : tensor<128x128xi1> to tensor<128x128xf32>
//       CHECK:  %[[rhs_mask_fp32:.+]] = tensorrt.identity %[[rhs_mask]] : tensor<128x128xi1> to tensor<128x128xf32>
//       CHECK:  %[[after_two_times:.+]] = tensorrt.element_wise <kPROD>(%[[lhs_mask_fp32]], %[[cst_f32_1]] : tensor<128x128xf32>, tensor<1x1xf32>) -> tensor<128x128xf32>
//       CHECK:  %[[after_minus_one:.+]] = tensorrt.element_wise <kSUB>(%[[after_two_times]], %[[cst_f32_0]] : tensor<128x128xf32>, tensor<1x1xf32>) -> tensor<128x128xf32>
//       CHECK:  %[[after_times_pi:.+]] = tensorrt.element_wise <kPROD>(%[[after_minus_one]], %[[cst_f32_2]] : tensor<128x128xf32>, tensor<1x1xf32>) -> tensor<128x128xf32>
//       CHECK:  %[[correction_term:.+]] = tensorrt.element_wise <kPROD>(%[[after_times_pi]], %[[rhs_mask_fp32]] : tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
//       CHECK:  %[[result:.+]] = tensorrt.element_wise <kSUB>(%[[intermediate_atan]], %[[correction_term]] : tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
//       CHECK:   return %[[result]] : tensor<128x128xf32>

// -----

func.func @hlo_expm1(%arg0: tensor<10x20x30xf32>) -> tensor<10x20x30xf32> {
    %0 = "stablehlo.exponential_minus_one"(%arg0) : (tensor<10x20x30xf32>) -> tensor<10x20x30xf32>
    return %0 : tensor<10x20x30xf32>
}

// CHECK-LABEL: @hlo_expm1
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10x20x30xf32>) -> tensor<10x20x30xf32>
//       CHECK: %[[cst_f32:.+]] = tensorrt.constant dense<1.000000e+00> : tensor<1x1x1xf32>
//       CHECK: %[[cst_f32_0:.+]] = tensorrt.constant dense<-1.000000e+00> : tensor<1x1x1xf32>
//       CHECK: %[[u:.+]] = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kEXP>} %[[arg0]] : tensor<10x20x30xf32>
//       CHECK: %[[u_eq_const_one:.+]] = tensorrt.element_wise <kEQUAL>(%[[u]], %[[cst_f32]] : tensor<10x20x30xf32>, tensor<1x1x1xf32>) -> tensor<10x20x30xi1>
//       CHECK: %[[u_eq_u:.+]] = tensorrt.element_wise <kEQUAL>(%[[u]], %[[u]] : tensor<10x20x30xf32>, tensor<10x20x30xf32>) -> tensor<10x20x30xi1>
//       CHECK: %[[u_eq_nan:.+]] = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kNOT>} %[[u_eq_u]] : tensor<10x20x30xi1>
//       CHECK: %[[u_eq_const_one_or_nan:.+]] = tensorrt.element_wise <kOR>(%[[u_eq_const_one]], %[[u_eq_nan]] : tensor<10x20x30xi1>, tensor<10x20x30xi1>) -> tensor<10x20x30xi1>
//       CHECK: %[[u_minus_const_one:.+]] = tensorrt.element_wise <kSUB>(%[[u]], %[[cst_f32]] : tensor<10x20x30xf32>, tensor<1x1x1xf32>) -> tensor<10x20x30xf32>
//       CHECK: %[[u_minus_const_one_equal_const_neg_one:.+]] = tensorrt.element_wise <kEQUAL>(%[[u_minus_const_one]], %[[cst_f32_0]] : tensor<10x20x30xf32>, tensor<1x1x1xf32>) -> tensor<10x20x30xi1>
//       CHECK: %[[log_u:.+]] = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kLOG>} %[[u]] : tensor<10x20x30xf32>
//       CHECK: %[[is_inf:.+]] = tensorrt.element_wise <kEQUAL>(%[[log_u]], %[[u]] : tensor<10x20x30xf32>, tensor<10x20x30xf32>) -> tensor<10x20x30xi1>
//       CHECK: %[[x_div_log_u:.+]]= tensorrt.element_wise <kDIV>(%[[arg0]], %[[log_u]] : tensor<10x20x30xf32>, tensor<10x20x30xf32>) -> tensor<10x20x30xf32>
//       CHECK: %[[expm1:.+]] = tensorrt.element_wise <kPROD>(%[[u_minus_const_one]], %[[x_div_log_u]]: tensor<10x20x30xf32>, tensor<10x20x30xf32>) -> tensor<10x20x30xf32>
//       CHECK: %[[expm1_pre:.+]] = tensorrt.select ins(%[[is_inf]], %[[u]], %[[expm1]] : tensor<10x20x30xi1>, tensor<10x20x30xf32>, tensor<10x20x30xf32>) -> tensor<10x20x30xf32>
//       CHECK: %[[check_minus_one_eq_const_neg_one:.+]] = tensorrt.select ins(%[[u_minus_const_one_equal_const_neg_one]], %[[cst_f32_0]], %[[expm1_pre]] : tensor<10x20x30xi1>, tensor<1x1x1xf32>, tensor<10x20x30xf32>) -> tensor<10x20x30xf32>
//       CHECK: %[[approx:.+]] = tensorrt.select ins(%[[u_eq_const_one_or_nan]], %[[arg0]], %[[check_minus_one_eq_const_neg_one]] : tensor<10x20x30xi1>, tensor<10x20x30xf32>, tensor<10x20x30xf32>) -> tensor<10x20x30xf32>

// -----

func.func @hlo_batch_norm_inference_f32(%input: tensor<2x3x224x224xf32>, %scale: tensor<3xf32>, %offset: tensor<3xf32>, %mean: tensor<3xf32>, %variance: tensor<3xf32>) -> tensor<2x3x224x224xf32> {
    %0 = "stablehlo.batch_norm_inference" (%input, %scale, %offset, %mean, %variance) {epsilon = 1.001000e-05 : f32, feature_index = 1 : i64} :
    (tensor<2x3x224x224xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<2x3x224x224xf32>
    return %0: tensor<2x3x224x224xf32>
}

// CHECK-LABEL: @hlo_batch_norm_inference_f32
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x3x224x224xf32>, %[[arg1:.+]]: tensor<3xf32>, %[[arg2:.+]]: tensor<3xf32>, %[[arg3:.+]]: tensor<3xf32>, %[[arg4:.+]]: tensor<3xf32>)
//       CHECK: %[[scale:.+]] = tensorrt.expand_rank %[[arg1]] : tensor<3xf32> to tensor<1x3x1x1xf32>
//       CHECK: %[[offset:.+]] = tensorrt.expand_rank %[[arg2]] : tensor<3xf32> to tensor<1x3x1x1xf32>
//       CHECK: %[[mean:.+]] = tensorrt.expand_rank %[[arg3]] : tensor<3xf32> to tensor<1x3x1x1xf32>
//       CHECK: %[[variance:.+]] = tensorrt.expand_rank %[[arg4]] : tensor<3xf32> to tensor<1x3x1x1xf32>
//       CHECK: %[[epsilon:.+]] = tensorrt.constant dense<1.001000e-05> : tensor<1x1x1x1xf32>
//       CHECK: %[[centeredOp:.+]] = tensorrt.element_wise <kSUB>(%[[arg0]], %[[mean]] : tensor<2x3x224x224xf32>, tensor<1x3x1x1xf32>) -> tensor<2x3x224x224xf32>
//       CHECK: %[[updatedVar:.+]] = tensorrt.element_wise <kSUM>(%[[variance]], %[[epsilon]] : tensor<1x3x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x1x1xf32>
//       CHECK: %[[stddev:.+]] = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kSQRT>} %[[updatedVar]] : tensor<1x3x1x1xf32>
//       CHECK: %[[normalizedOp:.+]] = tensorrt.element_wise <kDIV>(%[[centeredOp]], %[[stddev]] : tensor<2x3x224x224xf32>, tensor<1x3x1x1xf32>) -> tensor<2x3x224x224xf32>
//       CHECK: %[[scaledOp:.+]] = tensorrt.element_wise <kPROD>(%[[normalizedOp]], %[[scale]] : tensor<2x3x224x224xf32>, tensor<1x3x1x1xf32>) -> tensor<2x3x224x224xf32>
//       CHECK:  %[[shiftedOp:.+]] = tensorrt.element_wise <kSUM>(%[[scaledOp]], %[[offset]] : tensor<2x3x224x224xf32>, tensor<1x3x1x1xf32>) -> tensor<2x3x224x224xf32>


// -----

func.func @hlo_batch_norm_inference_f16(%input: tensor<2x3x224x224xf16>, %scale: tensor<3xf16>, %offset: tensor<3xf16>, %mean: tensor<3xf16>, %variance: tensor<3xf16>) -> tensor<2x3x224x224xf16> {
    %0 = "stablehlo.batch_norm_inference" (%input, %scale, %offset, %mean, %variance) {epsilon = 1.001000e-05 : f32, feature_index = 1 : i64} :
    (tensor<2x3x224x224xf16>, tensor<3xf16>, tensor<3xf16>, tensor<3xf16>, tensor<3xf16>) -> tensor<2x3x224x224xf16>
    return %0: tensor<2x3x224x224xf16>
}

// CHECK-LABEL: @hlo_batch_norm_inference_f16
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x3x224x224xf16>, %[[arg1:.+]]: tensor<3xf16>, %[[arg2:.+]]: tensor<3xf16>, %[[arg3:.+]]: tensor<3xf16>, %[[arg4:.+]]: tensor<3xf16>)
//       CHECK: %[[scale:.+]] = tensorrt.expand_rank %[[arg1]] : tensor<3xf16> to tensor<1x3x1x1xf16>
//       CHECK: %[[offset:.+]] = tensorrt.expand_rank %[[arg2]] : tensor<3xf16> to tensor<1x3x1x1xf16>
//       CHECK: %[[mean:.+]] = tensorrt.expand_rank %[[arg3]] : tensor<3xf16> to tensor<1x3x1x1xf16>
//       CHECK: %[[variance:.+]] = tensorrt.expand_rank %[[arg4]] : tensor<3xf16> to tensor<1x3x1x1xf16>
//       CHECK: %[[epsilon:.+]] = tensorrt.constant dense<1.001360e-05> : tensor<1x1x1x1xf16>
//       CHECK: %[[centeredOp:.+]] = tensorrt.element_wise <kSUB>(%[[arg0]], %[[mean]] : tensor<2x3x224x224xf16>, tensor<1x3x1x1xf16>) -> tensor<2x3x224x224xf16>
//       CHECK: %[[updatedVar:.+]] = tensorrt.element_wise <kSUM>(%[[variance]], %[[epsilon]] : tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>) -> tensor<1x3x1x1xf16>
//       CHECK: %[[stddev:.+]] = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kSQRT>} %[[updatedVar]] : tensor<1x3x1x1xf16>
//       CHECK: %[[normalizedOp:.+]] = tensorrt.element_wise <kDIV>(%[[centeredOp]], %[[stddev]] : tensor<2x3x224x224xf16>, tensor<1x3x1x1xf16>) -> tensor<2x3x224x224xf16>
//       CHECK: %[[scaledOp:.+]] = tensorrt.element_wise <kPROD>(%[[normalizedOp]], %[[scale]] : tensor<2x3x224x224xf16>, tensor<1x3x1x1xf16>) -> tensor<2x3x224x224xf16>
//       CHECK: %[[shiftedOp:.+]] = tensorrt.element_wise <kSUM>(%[[scaledOp]], %[[offset]] : tensor<2x3x224x224xf16>, tensor<1x3x1x1xf16>) -> tensor<2x3x224x224xf16>

// -----

func.func @hlo_batch_norm_inference_f32_feature_idx0(%input: tensor<4x3x3x3xf32>, %scale: tensor<4xf32>, %offset: tensor<4xf32>, %mean: tensor<4xf32>, %variance: tensor<4xf32>) -> tensor<4x3x3x3xf32> {
    %0 = "stablehlo.batch_norm_inference" (%input, %scale, %offset, %mean, %variance) {epsilon = 1.001000e-05 : f32, feature_index = 0 : i64} :
    (tensor<4x3x3x3xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4x3x3x3xf32>
    return %0: tensor<4x3x3x3xf32>
}

// CHECK-LABEL: @hlo_batch_norm_inference_f32_feature_idx0
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4x3x3x3xf32>, %[[arg1:.+]]: tensor<4xf32>, %[[arg2:.+]]: tensor<4xf32>, %[[arg3:.+]]: tensor<4xf32>, %[[arg4:.+]]: tensor<4xf32>)
//       CHECK: %[[scale:.+]] = tensorrt.expand_rank %[[arg1]] : tensor<4xf32> to tensor<4x1x1x1xf32>
//       CHECK: %[[offset:.+]] = tensorrt.expand_rank %[[arg2]] : tensor<4xf32> to tensor<4x1x1x1xf32>
//       CHECK: %[[mean:.+]] = tensorrt.expand_rank %[[arg3]] : tensor<4xf32> to tensor<4x1x1x1xf32>
//       CHECK: %[[variance:.+]] = tensorrt.expand_rank %[[arg4]] : tensor<4xf32> to tensor<4x1x1x1xf32>
//       CHECK: %[[epsilon:.+]] = tensorrt.constant dense<1.001000e-05> : tensor<1x1x1x1xf32>
//       CHECK: %[[centeredOp:.+]] = tensorrt.element_wise <kSUB>(%[[arg0]], %[[mean]] : tensor<4x3x3x3xf32>, tensor<4x1x1x1xf32>) -> tensor<4x3x3x3xf32>
//       CHECK: %[[updatedVar:.+]] = tensorrt.element_wise <kSUM>(%[[variance]], %[[epsilon]] : tensor<4x1x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<4x1x1x1xf32>
//       CHECK: %[[stddev:.+]] = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kSQRT>} %[[updatedVar]] : tensor<4x1x1x1xf32>
//       CHECK: %[[normalizedOp:.+]] = tensorrt.element_wise <kDIV>(%[[centeredOp]], %[[stddev]] : tensor<4x3x3x3xf32>, tensor<4x1x1x1xf32>) -> tensor<4x3x3x3xf32>
//       CHECK: %[[scaledOp:.+]] = tensorrt.element_wise <kPROD>(%[[normalizedOp]], %[[scale]] : tensor<4x3x3x3xf32>, tensor<4x1x1x1xf32>) -> tensor<4x3x3x3xf32>
//       CHECK: %[[shiftedOp:.+]] = tensorrt.element_wise <kSUM>(%[[scaledOp]], %[[offset]] : tensor<4x3x3x3xf32>, tensor<4x1x1x1xf32>) -> tensor<4x3x3x3xf32>

// -----

func.func @uniform_quantize(%arg: tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<i8:f32, 34.0>> {
  %0 = "stablehlo.uniform_quantize"(%arg) : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<i8:f32, 34.0>>
  return %0 : tensor<16x16x!quant.uniform<i8:f32, 34.0>>
}

// CHECK-LABEL: @uniform_quantize
//  CHECK-SAME: (%[[arg0:.+]]: tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<i8:f32, 3.400000e+01>> {
//       CHECK: %[[scale:.+]] = tensorrt.constant dense<3.400000e+01> : tensor<f32>
//       CHECK: %[[result:.+]] = tensorrt.quantize in(%[[arg0]] : tensor<16x16xf32>) scale(%[[scale]] : tensor<f32>) -> tensor<16x16x!quant.uniform<i8:f32, 3.400000e+01>>

// -----

func.func @uniform_dequantize(%arg: tensor<16x16x!quant.uniform<i8:f32, 34.0>>) -> tensor<16x16xf32> {
  %0 = "stablehlo.uniform_dequantize"(%arg) : (tensor<16x16x!quant.uniform<i8:f32, 34.0>>) -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

// CHECK-LABEL: @uniform_dequantize
//       CHECK: %[[scale:.+]] = tensorrt.constant dense<3.400000e+01> : tensor<f32>
//       CHECK: %[[result:.+]] = tensorrt.dequantize in(%[[arg0]] : tensor<16x16x!quant.uniform<i8:f32, 3.400000e+01>>) scale(%[[scale]] : tensor<f32>) -> tensor<16x16xf32>

// -----

func.func @op_dynamic_slice_1d(%arg0: tensor<16xf32>, %arg1: tensor<i32>) -> tensor<4xf32> {
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1) {
    slice_sizes = array<i64: 4>
  } : (tensor<16xf32>, tensor<i32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// CHECK-LABEL: @op_dynamic_slice_1d
//  CHECK-SAME:    (%[[arg0:.+]]: tensor<16xf32>, %[[arg1:.+]]: tensor<i32>) -> tensor<4xf32> {
//       CHECK:   %[[v1:.+]] = tensorrt.reshape %[[arg1]] : tensor<i32> to tensor<1xi32>
//       CHECK:   %[[v2:.+]] = tensorrt.slice %[[arg0]][%[[v1]]: tensor<1xi32>][4][1] : tensor<16xf32> to tensor<4xf32>
//       CHECK:   return %[[v2]] : tensor<4xf32>

// -----

func.func @op_dynamic_slice_2d(%arg0: tensor<16x16xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<4x4xf32> {
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) {
    slice_sizes = array<i64: 4, 4>
  } : (tensor<16x16xf32>, tensor<i32>, tensor<i32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @op_dynamic_slice_2d
//  CHECK-SAME:   (%[[arg0:.+]]: tensor<16x16xf32>, %[[arg1:.+]]: tensor<i32>, %[[arg2:.+]]:
//       CHECK:   %[[r0:.+]] = tensorrt.reshape %[[arg1]] : tensor<i32> to tensor<1xi32>
//       CHECK:   %[[r1:.+]] = tensorrt.reshape %[[arg2]] : tensor<i32> to tensor<1xi32>
//       CHECK:   %[[v2:.+]] = tensorrt.concatenation {axis = 0 : i32} ins(%[[r0]], %[[r1]] :
//       CHECK:   %[[v3:.+]] = tensorrt.slice %[[arg0]][%[[v2]]: tensor<2xi32>][4, 4][1, 1] : tensor<16x16xf32> to tensor<4x4xf32>
//       CHECK:   return %[[v3]]

// -----

func.func @hlo_log1p_fp16(%arg0 : tensor<10x20x30xf16>) -> tensor<10x20x30xf16> {
    %0 = "stablehlo.log_plus_one"(%arg0) : (tensor<10x20x30xf16>) -> tensor<10x20x30xf16>
    return %0 : tensor<10x20x30xf16>
}

// CHECK-LABEL: @hlo_log1p_fp16
//       CHECK: %[[v0:.+]] = tensorrt.identity %[[arg0:.+]] : tensor<10x20x30xf16> to tensor<10x20x30xf32>

// -----

func.func @hlo_expm1_fp16(%input : tensor<10x20x30xf16>) -> tensor<10x20x30xf16> {
    %0 = "stablehlo.exponential_minus_one"(%input) : (tensor<10x20x30xf16>) -> tensor<10x20x30xf16>
    return %0 : tensor<10x20x30xf16>
}

// CHECK-LABEL: @hlo_expm1_fp16
//       CHECK: %[[v0:.+]] = tensorrt.identity %[[arg0:.+]] : tensor<10x20x30xf16> to tensor<10x20x30xf32>

// -----

func.func @hlo_big_splat_constant() -> tensor<1024x1024xf32> {
  %0 = "stablehlo.constant"() {value = dense<1.0> : tensor<1024x1024xf32>} : () -> tensor<1024x1024xf32>
  return %0 : tensor<1024x1024xf32>
}

// CHECK-LABEL: @hlo_big_splat_constant
//  CHECK-SAME: () -> tensor<1024x1024xf32> {
//       CHECK:     %[[cst_f32:.+]] = tensorrt.constant dense<1.000000e+00> : tensor<1x1xf32>
//       CHECK:     %[[v0:.+]] = tensorrt.broadcast %[[cst_f32]] broadcast_dims<0, 1> : tensor<1x1xf32> to tensor<1024x1024xf32>
//       CHECK:     return %[[v0]] : tensor<1024x1024xf32>

// -----

func.func @hlo_big_splat_i1_constant() -> tensor<1024x1024x8xi1> {
  %0 = "stablehlo.constant"() {value = dense<1> : tensor<1024x1024x8xi1>} : () -> tensor<1024x1024x8xi1>
  return %0 : tensor<1024x1024x8xi1>
}
// CHECK-LABEL: @hlo_big_splat_i1_constant
//       CHECK:     %[[cst_i32:.+]] = tensorrt.constant dense<1> : tensor<1x1x1xi32>
//       CHECK:     %[[v0:.+]] = tensorrt.broadcast %[[cst_i32]] broadcast_dims<0, 1, 2> : tensor<1x1x1xi32> to tensor<1024x1024x8xi32>
//       CHECK:     %[[v1:.+]] = tensorrt.identity %[[v0]] : tensor<1024x1024x8xi32> to tensor<1024x1024x8xi1>
//       CHECK:     return %[[v1]] : tensor<1024x1024x8xi1>

// -----

func.func @hlo_big_splat_i1_constant2() -> tensor<1024x1024xi1> {
  %0 = "stablehlo.constant"() {value = dense<1> : tensor<1024x1024xi1>} : () -> tensor<1024x1024xi1>
  return %0 : tensor<1024x1024xi1>
}

// CHECK-LABEL: @hlo_big_splat_i1_constant2
//       CHECK:     %[[cst_i32:.+]] = tensorrt.constant dense<1> : tensor<1x1xi32>
//       CHECK:     %[[v0:.+]] = tensorrt.broadcast %[[cst_i32]] broadcast_dims<0, 1> : tensor<1x1xi32> to tensor<1024x1024xi32>
//       CHECK:     %[[v1:.+]] = tensorrt.identity %[[v0]] : tensor<1024x1024xi32> to tensor<1024x1024xi1>
//       CHECK:     return %[[v1]] : tensor<1024x1024xi1>

// -----

// Test that we don't raise errors on functions that don't need to be converted to TensorRT.
func.func @disregard_non_tensor_funcs(%arg0: i32) -> i32 {
  %0 = "unk.dialect"(%arg0, %arg0) : (i32, i32) -> i32
  return %0 : i32
}

// CHECK-LABEL: @disregard_non_tensor_funcs

// -----

func.func @dynamic_update_slice_conversion_1(%arg0: tensor<1x6x12x64xf32>) -> tensor<1x20x12x64xf32> {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = stablehlo.constant dense<0.000000e+00> : tensor<1x20x12x64xf32>
  %2 = stablehlo.dynamic_update_slice %1, %arg0, %0, %0, %0, %0 : (tensor<1x20x12x64xf32>, tensor<1x6x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
  return %2 : tensor<1x20x12x64xf32>
}

// CHECK-LABEL: @dynamic_update_slice_conversion_1
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x6x12x64xf32>)
//       CHECK:     %[[cst_i32:.+]] = tensorrt.constant dense<0>
//       CHECK:     %[[cst_f32:.+]] = tensorrt.constant dense<0.000000e+00>
//       CHECK:     %[[v0:.+]] = tensorrt.expand_rank %[[cst_i32]] : tensor<i32> to tensor<1xi32>
//       CHECK:     %[[cst_i32_0:.+]] = tensorrt.constant dense<1>
//       CHECK:     %[[cst_i32_1:.+]] = tensorrt.constant dense<[12, 64]>
//       CHECK:     %[[v1:.+]] = tensorrt.concatenation {axis = 0 : i32} ins(%[[cst_i32_0]], %[[v0]], %[[cst_i32_1]] :
//       CHECK:     %[[v2:.+]] = tensorrt.slice %[[cst_f32]][0, 0, 0, 0][%[[v1]]: tensor<4xi32>][1, 1, 1, 1]
//       CHECK:     %[[cst_i32_2:.+]] = tensorrt.constant dense<6>
//       CHECK:     %[[v3:.+]] = tensorrt.element_wise <kSUM>(%[[v0]], %[[cst_i32_2]] :
//       CHECK:     %[[cst_i32_3:.+]] = tensorrt.constant dense<0> : tensor<1xi32>
//       CHECK:     %[[cst_i32_4:.+]] = tensorrt.constant dense<0> : tensor<2xi32>
//       CHECK:     %[[v4:.+]] = tensorrt.concatenation {axis = 0 : i32} ins(%[[cst_i32_3]], %[[v3]], %[[cst_i32_4]]
//       CHECK:     %[[cst_i32_5:.+]] = tensorrt.constant dense<20>
//       CHECK:     %[[v5:.+]] = tensorrt.element_wise <kSUB>(%[[cst_i32_5]], %[[v3]] :
//       CHECK:     %[[cst_i32_6:.+]] = tensorrt.constant dense<1>
//       CHECK:     %[[cst_i32_7:.+]] = tensorrt.constant dense<[12, 64]>
//       CHECK:     %[[v6:.+]] = tensorrt.concatenation {axis = 0 : i32} ins(%[[cst_i32_6]], %[[v5]], %[[cst_i32_7]] :
//       CHECK:     %[[v7:.+]] = tensorrt.slice %[[cst_f32]][%[[v4]]: tensor<4xi32>][%[[v6]]: tensor<4xi32>][1, 1, 1, 1]
//       CHECK:     %[[v8:.+]] = tensorrt.concatenation {axis = 1 : i32} ins(%[[v2]], %[[arg0]], %[[v7]] :
//       CHECK:     return %[[v8]]

// -----

func.func @dynamic_update_slice_conversion_2(%arg0: tensor<1x20x12x64xf32>, %arg1: tensor<1x1x12x64xf32>, %arg2: tensor<i32>) -> tensor<1x20x12x64xf32> {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = stablehlo.dynamic_update_slice %arg0, %arg1, %0, %arg2, %0, %0 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
  return %1 : tensor<1x20x12x64xf32>
}

// CHECK-LABEL: @dynamic_update_slice_conversion_2
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x20x12x64xf32>, %[[arg1:.+]]: tensor<1x1x12x64xf32>, %[[arg2:.+]]: tensor<i32>)
//       CHECK:     %[[cst_i32:.+]] = tensorrt.constant dense<0> : tensor<i32>
//       CHECK:     %[[v0:.+]] = tensorrt.expand_rank %[[arg2]] : tensor<i32> to tensor<1xi32>
//       CHECK:     %[[cst_i32_0:.+]] = tensorrt.constant dense<1> : tensor<1xi32>
//       CHECK:     %[[cst_i32_1:.+]] = tensorrt.constant dense<[12, 64]> : tensor<2xi32>
//       CHECK:     %[[v1:.+]] = tensorrt.concatenation
//  CHECK-SAME:       axis = 0 : i32
//  CHECK-SAME:       ins(%[[cst_i32_0]], %[[v0]], %[[cst_i32_1]] : tensor<1xi32>, tensor<1xi32>, tensor<2xi32>)
//       CHECK:     %[[v2:.+]] = tensorrt.slice %[[arg0]][0, 0, 0, 0][%[[v1]]: tensor<4xi32>][1, 1, 1, 1] : tensor<1x20x12x64xf32> to tensor<1x?x12x64xf32>
//       CHECK:     %[[cst_i32_2:.+]] = tensorrt.constant dense<1> : tensor<1xi32>
//       CHECK:     %[[v3:.+]] = tensorrt.element_wise <kSUM>(%[[v0]], %[[cst_i32_2]] : tensor<1xi32>, tensor<1xi32>)
//       CHECK:     %[[cst_i32_3:.+]] = tensorrt.constant dense<0> : tensor<1xi32>
//       CHECK:     %[[cst_i32_4:.+]] = tensorrt.constant dense<0> : tensor<2xi32>
//       CHECK:     %[[v4:.+]] = tensorrt.concatenation
//  CHECK-SAME:        axis = 0 : i32
//  CHECK-SAME:        ins(%[[cst_i32_3]], %[[v3]], %[[cst_i32_4]] : tensor<1xi32>, tensor<1xi32>, tensor<2xi32>)
//       CHECK:     %[[cst_i32_5:.+]] = tensorrt.constant dense<20> : tensor<1xi32>
//       CHECK:     %[[v5:.+]] = tensorrt.element_wise <kSUB>(%[[cst_i32_5]], %[[v3]] : tensor<1xi32>, tensor<1xi32>)
//       CHECK:     %[[cst_i32_6:.+]] = tensorrt.constant dense<1> : tensor<1xi32>
//       CHECK:     %[[cst_i32_7:.+]] = tensorrt.constant dense<[12, 64]> : tensor<2xi32>
//       CHECK:     %[[v6:.+]] = tensorrt.concatenation
//  CHECK-SAME:       axis = 0 : i32
//  CHECK-SAME:       ins(%[[cst_i32_6]], %[[v5]], %[[cst_i32_7]] : tensor<1xi32>, tensor<1xi32>, tensor<2xi32>)
//       CHECK:     %[[v7:.+]] = tensorrt.slice %[[arg0]][%[[v4]]: tensor<4xi32>][%[[v6]]: tensor<4xi32>][1, 1, 1, 1] : tensor<1x20x12x64xf32> to tensor<1x?x12x64xf32>
//       CHECK:     %[[v8:.+]] = tensorrt.concatenation
//  CHECK-SAME:       axis = 1 : i32
//  CHECK-SAME:       ins(%[[v2]], %[[arg1]], %[[v7]] : tensor<1x?x12x64xf32>, tensor<1x1x12x64xf32>, tensor<1x?x12x64xf32>)
//       CHECK:     return %[[v8]]

// -----

func.func @dynamic_update_slice_conversion_unsupported1(%arg0: tensor<1x20x12x64xf32>, %arg1: tensor<1x1x1x64xf32>, %arg2: tensor<i32>) -> tensor<1x20x12x64xf32> {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = stablehlo.dynamic_update_slice %arg0, %arg1, %0, %arg2, %arg2, %0 : (tensor<1x20x12x64xf32>, tensor<1x1x1x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
  return %1 : tensor<1x20x12x64xf32>
}

// CHECK-LABEL: @dynamic_update_slice_conversion_unsupported1
//       CHECK:  stablehlo.dynamic_update_slice

// -----

func.func @dynamic_update_slice_conversion_unsupported2(%arg0: tensor<1x20x12x64xf32>, %arg1: tensor<1x1x1x64xf32>, %arg2: tensor<i32>) -> tensor<1x20x12x64xf32> {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = stablehlo.dynamic_update_slice %arg0, %arg1, %0, %arg2, %0, %0 : (tensor<1x20x12x64xf32>, tensor<1x1x1x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
  return %1 : tensor<1x20x12x64xf32>
}

// CHECK-LABEL: @dynamic_update_slice_conversion_unsupported2
//       CHECK:  stablehlo.dynamic_update_slice

// -----

func.func @scatter_slice_update(%arg0: tensor<1x134xi32>, %arg1: tensor<1x2xi32>, %arg2: tensor<1x1x5xi32>) -> tensor<1x134xi32> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    stablehlo.return %arg4 : tensor<i32>
  }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false} : (tensor<1x134xi32>, tensor<1x2xi32>, tensor<1x1x5xi32>) -> tensor<1x134xi32>
  return %0 : tensor<1x134xi32>
}

// CHECK-LABEL: @scatter_slice_update
//       CHECK:     %[[v0:.+]] = tensorrt.slice %arg1[0, 1][1, 1][1, 1] : tensor<1x2xi32> to tensor<1x1xi32>
//       CHECK:     %[[v1:.+]] = tensorrt.collapse_rank %[[v0]] : tensor<1x1xi32> to tensor<i32>
//       CHECK:     %[[v2:.+]] = tensorrt.collapse_rank %arg2 : tensor<1x1x5xi32> to tensor<1x5xi32>
//       CHECK:     %[[v3:.+]] = tensorrt.constant dense<1> : tensor<2xi32>
//       CHECK:     %[[v4:.+]] = tensorrt.linspace[%[[v1]] : tensor<i32>] [ static] [%[[v3]] : tensor<2xi32>] : tensor<1x5xi32>
//       CHECK:     %[[v5:.+]] = tensorrt.scatter_elements {axis = 1 : i64} data(%arg0 : tensor<1x134xi32>) indices(%[[v4]] : tensor<1x5xi32>) updates(%[[v2]] : tensor<1x5xi32>)
//       CHECK:     return %[[v5]] : tensor<1x134xi32>

// -----

func.func @scatter_slice_update_f16(%arg0: tensor<1x134xf16>, %arg1: tensor<1x2xi32>, %arg2: tensor<1x1x5xf16>) -> tensor<1x134xf16> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):
    stablehlo.return %arg4 : tensor<f16>
  }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false} : (tensor<1x134xf16>, tensor<1x2xi32>, tensor<1x1x5xf16>) -> tensor<1x134xf16>
  return %0 : tensor<1x134xf16>
}

// CHECK-LABEL: @scatter_slice_update_f16
//       CHECK:     %[[v0:.+]] = tensorrt.slice %arg1[0, 1][1, 1][1, 1] : tensor<1x2xi32> to tensor<1x1xi32>
//       CHECK:     %[[v1:.+]] = tensorrt.collapse_rank %[[v0]] : tensor<1x1xi32> to tensor<i32>
//       CHECK:     %[[v2:.+]] = tensorrt.collapse_rank %arg2 : tensor<1x1x5xf16> to tensor<1x5xf16>
//       CHECK:     %[[v3:.+]] = tensorrt.constant dense<1> : tensor<2xi32>
//       CHECK:     %[[v4:.+]] = tensorrt.linspace[%[[v1]] : tensor<i32>] [ static] [%[[v3]] : tensor<2xi32>] : tensor<1x5xi32>
//       CHECK:     %[[v5:.+]] = tensorrt.scatter_elements {axis = 1 : i64} data(%arg0 : tensor<1x134xf16>) indices(%[[v4]] : tensor<1x5xi32>) updates(%[[v2]] : tensor<1x5xf16>)
//       CHECK:     return %[[v5]] : tensor<1x134xf16>

// -----

func.func @scatter_slice_update_i1(%arg0: tensor<1x134xi1>, %arg1: tensor<1x2xi32>, %arg2: tensor<1x1x5xi1>) -> tensor<1x134xi1> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<i1>, %arg4: tensor<i1>):
    stablehlo.return %arg4 : tensor<i1>
  }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false} : (tensor<1x134xi1>, tensor<1x2xi32>, tensor<1x1x5xi1>) -> tensor<1x134xi1>
  return %0 : tensor<1x134xi1>
}

// CHECK-LABEL: @scatter_slice_update_i1
//       CHECK:     %[[v0:.+]] = tensorrt.slice %arg1[0, 1][1, 1][1, 1] : tensor<1x2xi32> to tensor<1x1xi32>
//       CHECK:     %[[v1:.+]] = tensorrt.collapse_rank %[[v0]] : tensor<1x1xi32> to tensor<i32>
//       CHECK:     %[[v2:.+]] = tensorrt.collapse_rank %arg2 : tensor<1x1x5xi1> to tensor<1x5xi1>
//       CHECK:     %[[v3:.+]] = tensorrt.constant dense<1> : tensor<2xi32>
//       CHECK:     %[[v4:.+]] = tensorrt.linspace[%[[v1]] : tensor<i32>] [ static] [%[[v3]] : tensor<2xi32>] : tensor<1x5xi32>
//       CHECK:     %[[v5:.+]] = tensorrt.identity %arg0 : tensor<1x134xi1> to tensor<1x134xi32>
//       CHECK:     %[[v6:.+]] = tensorrt.identity %[[v2]] : tensor<1x5xi1> to tensor<1x5xi32>
//       CHECK:     %[[v7:.+]] = tensorrt.scatter_elements {axis = 1 : i64} data(%[[v5]] : tensor<1x134xi32>) indices(%[[v4]] : tensor<1x5xi32>) updates(%[[v6]] : tensor<1x5xi32>)
//       CHECK:     %[[v8:.+]] = tensorrt.identity %[[v7]] : tensor<1x134xi32> to tensor<1x134xi1>
//       CHECK:     return %[[v8]] : tensor<1x134xi1>

// -----

func.func @quantize_pt_to_i8_static(%arg0: tensor<2x3x300x300xf32>) -> tensor<2x3x300x300xi8> {
  %0 = stablehlo.composite "tensorrt.pt_q" %arg0 {composite_attributes = {axis = -1 : i32, is_pointwise, scale = dense<8.000000e-01> : tensor<f32>}, decomposition = @pt_q} : (tensor<2x3x300x300xf32>) -> tensor<2x3x300x300xi8>
  return %0 : tensor<2x3x300x300xi8>
}
func.func private @pt_q(%arg0: tensor<2x3x300x300xf32>) -> tensor<2x3x300x300xi8> attributes {plan.decomposition} {
  %cst = stablehlo.constant dense<-1.280000e+02> : tensor<f32>
  %cst_0 = stablehlo.constant dense<1.270000e+02> : tensor<f32>
  %cst_1 = stablehlo.constant dense<8.000000e-01> : tensor<f32>
  %0 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<2x3x300x300xf32>
  %1 = stablehlo.divide %arg0, %0 : tensor<2x3x300x300xf32>
  %2 = stablehlo.round_nearest_even %1 : tensor<2x3x300x300xf32>
  %3 = stablehlo.clamp %cst, %2, %cst_0 : (tensor<f32>, tensor<2x3x300x300xf32>, tensor<f32>) -> tensor<2x3x300x300xf32>
  %4 = stablehlo.convert %3 : (tensor<2x3x300x300xf32>) -> tensor<2x3x300x300xi8>
  return %4 : tensor<2x3x300x300xi8>
}

// CHECK-LABEL: quantize_pt_to_i8_static
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x3x300x300xf32>) -> tensor<2x3x300x300xi8>
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.constant dense<8.000000e-01> : tensor<f32>
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.quantize in(%[[arg0]] : tensor<2x3x300x300xf32>) scale(%[[v0]] : tensor<f32>) -> tensor<2x3x300x300xi8>
//  CHECK-NEXT: return %[[v1]] : tensor<2x3x300x300xi8>

// -----

func.func @quantize_pc_to_i8_dynamic(%arg0: tensor<?x3x?x?xf32>) -> tensor<?x3x?x?xi8> {
  %0 = stablehlo.composite "tensorrt.pc_q" %arg0 {composite_attributes = {axis = 1 : i32, is_pointwise, scale = dense_resource<__elided__> : tensor<3xf32>}, decomposition = @pc_q} : (tensor<?x3x?x?xf32>) -> tensor<?x3x?x?xi8>
  return %0 : tensor<?x3x?x?xi8>
}
func.func private @pc_q(%arg0: tensor<?x3x?x?xf32>) -> tensor<?x3x?x?xi8> attributes {plan.decomposition} {
  %cst = stablehlo.constant dense<-1.280000e+02> : tensor<f32>
  %cst_0 = stablehlo.constant dense<1.270000e+02> : tensor<f32>
  %cst_1 = stablehlo.constant dense_resource<__elided__> : tensor<3xf32>
  %c = stablehlo.constant dense<3> : tensor<1xi32>
  %0 = stablehlo.get_dimension_size %arg0, dim = 3 : (tensor<?x3x?x?xf32>) -> tensor<i32>
  %1 = stablehlo.reshape %0 : (tensor<i32>) -> tensor<1xi32>
  %2 = stablehlo.get_dimension_size %arg0, dim = 2 : (tensor<?x3x?x?xf32>) -> tensor<i32>
  %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
  %4 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x3x?x?xf32>) -> tensor<i32>
  %5 = stablehlo.reshape %4 : (tensor<i32>) -> tensor<1xi32>
  %6 = stablehlo.concatenate %5, %c, %3, %1, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  %7 = stablehlo.dynamic_broadcast_in_dim %cst_1, %6, dims = [1] : (tensor<3xf32>, tensor<4xi32>) -> tensor<?x3x?x?xf32>
  %8 = stablehlo.divide %arg0, %7 : tensor<?x3x?x?xf32>
  %9 = stablehlo.round_nearest_even %8 : tensor<?x3x?x?xf32>
  %10 = stablehlo.clamp %cst, %9, %cst_0 : (tensor<f32>, tensor<?x3x?x?xf32>, tensor<f32>) -> tensor<?x3x?x?xf32>
  %11 = stablehlo.convert %10 : (tensor<?x3x?x?xf32>) -> tensor<?x3x?x?xi8>
  return %11 : tensor<?x3x?x?xi8>
}

// CHECK-LABEL: quantize_pc_to_i8_dynamic
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x3x?x?xf32>) -> tensor<?x3x?x?xi8>
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.constant dense_resource<__elided__> : tensor<3xf32>
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.quantize {axis = 1 : i32} in(%[[arg0]] : tensor<?x3x?x?xf32>) scale(%[[v0]] : tensor<3xf32>) -> tensor<?x3x?x?xi8>
//  CHECK-NEXT: return %[[v1]] : tensor<?x3x?x?xi8>

// -----

func.func @quantize_pc_to_i8_eager() -> tensor<258x256xi8> {
  %cst = stablehlo.constant dense<1.000000e+00> : tensor<258x256xf32>
  %0 = stablehlo.composite "tensorrt.pc_q" %cst {composite_attributes = {axis = 1 : i32, is_pointwise, scale = dense_resource<__elided__> : tensor<256xf32>}, decomposition = @pc_q} : (tensor<258x256xf32>) -> tensor<258x256xi8>
  return %0 : tensor<258x256xi8>
}
func.func private @pc_q(%arg0: tensor<258x256xf32>) -> tensor<258x256xi8> attributes {plan.decomposition} {
  %cst = stablehlo.constant dense<-1.280000e+02> : tensor<f32>
  %cst_0 = stablehlo.constant dense<1.270000e+02> : tensor<f32>
  %cst_1 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
  %0 = stablehlo.broadcast_in_dim %cst_1, dims = [1] : (tensor<256xf32>) -> tensor<258x256xf32>
  %1 = stablehlo.divide %arg0, %0 : tensor<258x256xf32>
  %2 = stablehlo.round_nearest_even %1 : tensor<258x256xf32>
  %3 = stablehlo.clamp %cst, %2, %cst_0 : (tensor<f32>, tensor<258x256xf32>, tensor<f32>) -> tensor<258x256xf32>
  %4 = stablehlo.convert %3 : (tensor<258x256xf32>) -> tensor<258x256xi8>
  return %4 : tensor<258x256xi8>
}

// CHECK-LABEL: quantize_pc_to_i8_eager
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.constant dense<1.000000e+00> : tensor<1x1xf32>
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.broadcast %[[v0]] broadcast_dims<0, 1> : tensor<1x1xf32> to tensor<258x256xf32>
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.constant dense_resource<__elided__> : tensor<256xf32>
//  CHECK-NEXT: %[[v3:.+]] = tensorrt.quantize {axis = 1 : i32} in(%[[v1]] : tensor<258x256xf32>) scale(%[[v2]] : tensor<256xf32>) -> tensor<258x256xi8>
//  CHECK-NEXT: return %[[v3]] : tensor<258x256xi8>

// -----

func.func @large_weight() -> tensor<258x256xf32> {
  %c = stablehlo.constant dense_resource<__elided__> : tensor<258x256xi4>
  %0 = stablehlo.composite "tensorrt.block_dq" %c {composite_attributes = {axis = -1 : i32, is_pointwise, scale = dense_resource<__elided__> : tensor<2x256xf32>}, decomposition = @block_dq} : (tensor<258x256xi4>) -> tensor<258x256xf32>
  return %0 : tensor<258x256xf32>
}
func.func private @block_dq(%arg0: tensor<258x256xi4>) -> tensor<258x256xf32> attributes {plan.decomposition} {
  %cst = stablehlo.constant dense_resource<__elided__> : tensor<2x256xf32>
  %0 = stablehlo.broadcast_in_dim %cst, dims = [1, 2] : (tensor<2x256xf32>) -> tensor<129x2x256xf32>
  %1 = stablehlo.reshape %0 : (tensor<129x2x256xf32>) -> tensor<258x256xf32>
  %2 = stablehlo.convert %arg0 : (tensor<258x256xi4>) -> tensor<258x256xf32>
  %3 = stablehlo.multiply %2, %1 : tensor<258x256xf32>
  return %3 : tensor<258x256xf32>
}

// CHECK-LABEL: large_weight
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.constant dense_resource<__elided__> : tensor<258x256xi4>
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.constant dense_resource<__elided__> : tensor<2x256xf32>
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.dequantize in(%[[v0]] : tensor<258x256xi4>) scale(%[[v1]] : tensor<2x256xf32>) -> tensor<258x256xf32>
//  CHECK-NEXT: return %[[v2]] : tensor<258x256xf32>

// -----

func.func @quantize_pt_bf16_to_fp8_static() -> tensor<2xf8E4M3FN> {
  %cst = stablehlo.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xbf16>
  %0 = stablehlo.composite "tensorrt.pt_q" %cst {composite_attributes = {axis = -1 : i32, is_pointwise, scale = dense<5.000000e-01> : tensor<bf16>}, decomposition = @pt_q} : (tensor<2xbf16>) -> tensor<2xf8E4M3FN>
  return %0 : tensor<2xf8E4M3FN>
}
func.func private @pt_q(%arg0: tensor<2xbf16>) -> tensor<2xf8E4M3FN> attributes {plan.decomposition} {
  %cst = stablehlo.constant dense<-4.480000e+02> : tensor<f32>
  %cst_0 = stablehlo.constant dense<4.480000e+02> : tensor<f32>
  %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
  %0 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<bf16>) -> tensor<2xbf16>
  %1 = stablehlo.divide %arg0, %0 : tensor<2xbf16>
  %2 = stablehlo.round_nearest_even %1 : tensor<2xbf16>
  %3 = stablehlo.convert %cst_0 : (tensor<f32>) -> tensor<bf16>
  %4 = stablehlo.convert %cst : (tensor<f32>) -> tensor<bf16>
  %5 = stablehlo.clamp %4, %2, %3 : (tensor<bf16>, tensor<2xbf16>, tensor<bf16>) -> tensor<2xbf16>
  %6 = stablehlo.convert %5 : (tensor<2xbf16>) -> tensor<2xf8E4M3FN>
  return %6 : tensor<2xf8E4M3FN>
}

// CHECK-LABEL: quantize_pt_bf16_to_fp8_static
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xbf16>
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.constant dense<5.000000e-01> : tensor<bf16>
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.quantize in(%[[v0]] : tensor<2xbf16>) scale(%[[v1]] : tensor<bf16>) -> tensor<2xf8E4M3FN>
//  CHECK-NEXT: return %[[v2]] : tensor<2xf8E4M3FN>

// -----

func.func @compare_boolean_inputs(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i1> {
  %0 = stablehlo.constant dense<1> : tensor<i32>
  %1 = stablehlo.compare  LT, %arg0, %0 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %2 = stablehlo.compare  LT, %arg1, %0 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %3 = stablehlo.compare  NE, %1, %2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
  return %3 : tensor<i1>
}

// CHECK-LABEL: @compare_boolean_inputs
//       CHECK:  %[[v0:.+]] = tensorrt.element_wise <kLESS>
//  CHECK-SAME:   tensor<i32>, tensor<i32>) -> tensor<i1>
//       CHECK:  %[[v1:.+]] = tensorrt.element_wise <kLESS>
//  CHECK-SAME:   tensor<i32>, tensor<i32>) -> tensor<i1>
//       CHECK: %[[v2:.+]] = tensorrt.identity %[[v0]] : tensor<i1> to tensor<i32>
//       CHECK: %[[v3:.+]] = tensorrt.identity %[[v1]] : tensor<i1> to tensor<i32>
//       CHECK: tensorrt.element_wise <kEQUAL>(%[[v2]], %[[v3]] : tensor<i32>, tensor<i32>) -> tensor<i1>
//       CHECK: tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kNOT>}

// -----

func.func @jnp_cumsum_2d_i32(%arg0: tensor<1x134xi32>) -> tensor<1x134xi32> {
  %cst = arith.constant dense<0> : tensor<i32>
  %4 = "stablehlo.reduce_window"(%arg0, %cst) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [133, 0]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 134>, window_strides = array<i64: 1, 1>}> ({
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %5 = stablehlo.add %arg1, %arg2 : tensor<i32>
    stablehlo.return %5 : tensor<i32>
  }) : (tensor<1x134xi32>, tensor<i32>) -> tensor<1x134xi32>
  return %4 : tensor<1x134xi32>
}

// CHECK-LABEL: @jnp_cumsum_2d_i32
//       CHECK:  %[[v0:.+]] = tensorrt.identity %arg0
//       CHECK:  %[[v1:.+]] = tensorrt.expand_rank %[[v0]] : tensor<1x134xf32> to tensor<1x1x1x134xf32>
//       CHECK:  %[[v2:.+]] = tensorrt.constant dense<1.000000e+00> : tensor<1x1x1x134xf32>
//       CHECK:  %[[v3:.+]] = tensorrt.convolution
//       CHECK-SAME: post_padding = array<i64: 0, 0>
//       CHECK-SAME: pre_padding = array<i64: 0, 133>
//       CHECK-SAME: in(%[[v1]] : tensor<1x1x1x134xf32>) kernel(%[[v2]] : tensor<1x1x1x134xf32>) -> tensor<1x1x1x134xf32>
//       CHECK:  %[[v4:.+]] = tensorrt.identity %[[v3]] : tensor<1x1x1x134xf32> to tensor<1x1x1x134xi32>
//       CHECK:  %[[v5:.+]] = tensorrt.reshape %[[v4]] : tensor<1x1x1x134xi32> to tensor<1x134xi32>

// -----

func.func @jnp_cumsum_1d_i32(%arg0: tensor<134xi32>) -> tensor<134xi32> {
  %cst = arith.constant dense<0> : tensor<i32>
  %4 = "stablehlo.reduce_window"(%arg0, %cst) <{base_dilations = array<i64: 1>, padding = dense<[[133, 0]]> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 134>, window_strides = array<i64: 1>}> ({
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %5 = stablehlo.add %arg1, %arg2 : tensor<i32>
    stablehlo.return %5 : tensor<i32>
  }) : (tensor<134xi32>, tensor<i32>) -> tensor<134xi32>
  return %4 : tensor<134xi32>
}

// CHECK-LABEL: @jnp_cumsum_1d_i32
//       CHECK:  %[[v0:.+]] = tensorrt.identity %arg0
//       CHECK:  %[[v1:.+]] = tensorrt.expand_rank %[[v0]] : tensor<134xf32> to tensor<1x1x1x134xf32>
//       CHECK:  %[[v2:.+]] = tensorrt.constant dense<1.000000e+00> : tensor<1x1x1x134xf32>
//       CHECK:  %[[v3:.+]] = tensorrt.convolution
//       CHECK-SAME: post_padding = array<i64: 0, 0>
//       CHECK-SAME: pre_padding = array<i64: 0, 133>
//       CHECK-SAME: in(%[[v1]] : tensor<1x1x1x134xf32>) kernel(%[[v2]] : tensor<1x1x1x134xf32>) -> tensor<1x1x1x134xf32>
//       CHECK:  %[[v4:.+]] = tensorrt.identity %[[v3]] : tensor<1x1x1x134xf32> to tensor<1x1x1x134xi32>
//       CHECK:  %[[v5:.+]] = tensorrt.reshape %[[v4]] : tensor<1x1x1x134xi32> to tensor<134xi32>

// -----

func.func @jnp_cumsum_2d_axis0_i32(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %cst = arith.constant dense<0> : tensor<i32>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{base_dilations = array<i64: 1, 1>, padding = dense<[[1, 0], [0, 0]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 2, 1>, window_strides = array<i64: 1, 1>}> ({
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
    stablehlo.return %1 : tensor<i32>
  }) : (tensor<2x2xi32>, tensor<i32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

// CHECK-LABEL: @jnp_cumsum_2d_axis0_i32
//       CHECK:  %[[v0:.+]] = tensorrt.identity %arg0
//       CHECK:  %[[v1:.+]] = tensorrt.expand_rank %[[v0]] : tensor<2x2xf32> to tensor<1x1x2x2xf32>
//       CHECK:  %[[v2:.+]] = tensorrt.constant dense<1.000000e+00> : tensor<1x1x2x1xf32>
//       CHECK:  %[[v3:.+]] = tensorrt.convolution
//       CHECK-SAME: post_padding = array<i64: 0, 0>
//       CHECK-SAME: pre_padding = array<i64: 1, 0>
//       CHECK-SAME: in(%[[v1]] : tensor<1x1x2x2xf32>) kernel(%[[v2]] :  tensor<1x1x2x1xf32>) -> tensor<1x1x2x2xf32>
//       CHECK:  %[[v4:.+]] = tensorrt.identity %[[v3]] : tensor<1x1x2x2xf32> to tensor<1x1x2x2xi32>
//       CHECK:  %[[v5:.+]] = tensorrt.reshape %[[v4]] : tensor<1x1x2x2xi32> to tensor<2x2xi32>

// -----

func.func @jnp_cumsum_3d_i32(%arg0: tensor<1x2x2xi32>) -> tensor<1x2x2xi32> {
  %cst = arith.constant dense<0> : tensor<i32>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [1, 0]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 2>, window_strides = array<i64: 1, 1, 1>}> ({
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
    stablehlo.return %1 : tensor<i32>
  }) : (tensor<1x2x2xi32>, tensor<i32>) -> tensor<1x2x2xi32>
  return %0 : tensor<1x2x2xi32>
}

// CHECK-LABEL: @jnp_cumsum_3d_i32
//       CHECK:  %[[v0:.+]] = tensorrt.identity %arg0
//       CHECK:  %[[v1:.+]] = tensorrt.expand_rank %[[v0]] : tensor<1x2x2xf32> to tensor<1x1x1x2x2xf32>
//       CHECK:  %[[v2:.+]] = tensorrt.constant dense<1.000000e+00> : tensor<1x1x1x1x2xf32>
//       CHECK:  %[[v3:.+]] = tensorrt.convolution
//       CHECK-SAME: post_padding = array<i64: 0, 0, 0>
//       CHECK-SAME: pre_padding = array<i64: 0, 0, 1>
//       CHECK-SAME: in(%[[v1]] : tensor<1x1x1x2x2xf32>) kernel(%[[v2]] : tensor<1x1x1x1x2xf32>) -> tensor<1x1x1x2x2xf32>
//       CHECK:  %[[v4:.+]] = tensorrt.identity %[[v3]] : tensor<1x1x1x2x2xf32> to tensor<1x1x1x2x2xi32>
//       CHECK:  %[[v5:.+]] = tensorrt.reshape %[[v4]] : tensor<1x1x1x2x2xi32> to tensor<1x2x2xi32>

// -----

func.func @jnp_cumsum_2d_f32(%arg0: tensor<1x134xf32>) -> tensor<1x134xf32> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [133, 0]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 134>, window_strides = array<i64: 1, 1>}> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }) : (tensor<1x134xf32>, tensor<f32>) -> tensor<1x134xf32>
  return %0 : tensor<1x134xf32>
}

// CHECK-LABEL: @jnp_cumsum_2d_f32
//       CHECK:  %[[v1:.+]] = tensorrt.expand_rank %arg0 : tensor<1x134xf32> to tensor<1x1x1x134xf32>
//       CHECK:  %[[v2:.+]] = tensorrt.constant dense<1.000000e+00> : tensor<1x1x1x134xf32>
//       CHECK:  %[[v3:.+]] = tensorrt.convolution
//       CHECK-SAME: post_padding = array<i64: 0, 0>
//       CHECK-SAME: pre_padding = array<i64: 0, 133>
//       CHECK-SAME: in(%[[v1]] : tensor<1x1x1x134xf32>) kernel(%[[v2]] : tensor<1x1x1x134xf32>) -> tensor<1x1x1x134xf32>
//       CHECK:  %[[v4:.+]] = tensorrt.reshape %[[v3]] : tensor<1x1x1x134xf32> to tensor<1x134xf32>

// -----

func.func @jnp_cumsum_2d_f16(%arg0: tensor<1x134xf16>) -> tensor<1x134xf16> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [133, 0]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 134>, window_strides = array<i64: 1, 1>}> ({
  ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):
    %1 = stablehlo.add %arg1, %arg2 : tensor<f16>
    stablehlo.return %1 : tensor<f16>
  }) : (tensor<1x134xf16>, tensor<f16>) -> tensor<1x134xf16>
  return %0 : tensor<1x134xf16>
}

// CHECK-LABEL: @jnp_cumsum_2d_f16
//       CHECK:  %[[v1:.+]] = tensorrt.expand_rank %arg0 : tensor<1x134xf16> to tensor<1x1x1x134xf16>
//       CHECK:  %[[v2:.+]] = tensorrt.constant dense<1.000000e+00> : tensor<1x1x1x134xf16>
//       CHECK:  %[[v3:.+]] = tensorrt.convolution
//       CHECK-SAME: post_padding = array<i64: 0, 0>
//       CHECK-SAME: pre_padding = array<i64: 0, 133>
//       CHECK-SAME: in(%[[v1]] : tensor<1x1x1x134xf16>) kernel(%[[v2]] : tensor<1x1x1x134xf16>) -> tensor<1x1x1x134xf16>
//       CHECK:  %[[v4:.+]] = tensorrt.reshape %[[v3]] : tensor<1x1x1x134xf16> to tensor<1x134xf16>
