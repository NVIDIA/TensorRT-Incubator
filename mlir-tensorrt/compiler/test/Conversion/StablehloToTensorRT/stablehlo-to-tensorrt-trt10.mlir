// RUN: mlir-tensorrt-opt -split-input-file %s --convert-stablehlo-to-tensorrt="trt-major-version=10" -allow-unregistered-dialect | FileCheck %s

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

func.func @hlo_dynamic_iota_0(%arg0 : tensor<1xi32>) -> tensor<?xi32> {
  %0 = "stablehlo.dynamic_iota"(%arg0) {iota_dimension = 0 : i64} : (tensor<1xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK-LABEL: @hlo_dynamic_iota_0
//       CHECK:  tensorrt.linspace
//  CHECK-SAME:    [ 0.00{{.+}}] [%arg0 : tensor<1xi32>] [ 1.000{{.+}}] : tensor<?xi32>

// -----

func.func @dynamic_nd_iota_1(%arg0 : tensor<2xi32>) -> tensor<?x3xi32> {
  %0 = "stablehlo.dynamic_iota"(%arg0) {iota_dimension = 1 : i64} : (tensor<2xi32>) -> tensor<?x3xi32>
  return %0 : tensor<?x3xi32>
}

// CHECK-LABEL: func.func @dynamic_nd_iota_1
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2xi32>) -> tensor<?x3xi32> {
//       CHECK:     %[[cst_i32:.+]] = tensorrt.constant dense<0> : tensor<i32>
//       CHECK:     %[[cst_i32_0:.+]] = tensorrt.constant dense<[0, 1]> : tensor<2xi32>
//       CHECK:     %[[v0:.+]] = tensorrt.linspace[%[[cst_i32]] : tensor<i32>] [%[[arg0]] : tensor<2xi32>] [%[[cst_i32_0]] : tensor<2xi32>] : tensor<?x3xi32>
//       CHECK:     return %[[v0]] : tensor<?x3xi32>

// -----

func.func @dynamic_nd_iota_2(%arg0 : tensor<2xi32>) -> tensor<?x3xi32> {
  %0 = "stablehlo.dynamic_iota"(%arg0) {iota_dimension = 0 : i64} : (tensor<2xi32>) -> tensor<?x3xi32>
  return %0 : tensor<?x3xi32>
}

// CHECK-LABEL: func.func @dynamic_nd_iota_2
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2xi32>) -> tensor<?x3xi32> {
//       CHECK:     %[[cst_i32:.+]] = tensorrt.constant dense<0> : tensor<i32>
//       CHECK:     %[[cst_i32_0:.+]] = tensorrt.constant dense<[1, 0]> : tensor<2xi32>
//       CHECK:     %[[v0:.+]] = tensorrt.linspace[%[[cst_i32]] : tensor<i32>] [%[[arg0]] : tensor<2xi32>] [%[[cst_i32_0]] : tensor<2xi32>] : tensor<?x3xi32>
//       CHECK:     return %[[v0]] : tensor<?x3xi32>

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

func.func @scatter_slice_update_f16_axis1(%arg0: tensor<1x134xf16>, %arg1: tensor<1x2xi32>, %arg2: tensor<1x1x5xf16>) -> tensor<1x134xf16> {
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

func.func @scatter_slice_update_i1_axis1(%arg0: tensor<1x134xi1>, %arg1: tensor<1x2xi32>, %arg2: tensor<1x1x5xi1>) -> tensor<1x134xi1> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<i1>, %arg4: tensor<i1>):
    stablehlo.return %arg4 : tensor<i1>
  }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false} : (tensor<1x134xi1>, tensor<1x2xi32>, tensor<1x1x5xi1>) -> tensor<1x134xi1>
  return %0 : tensor<1x134xi1>
}

// CHECK-LABEL: @scatter_slice_update_i1_axis1
//       CHECK:     %[[v0:.+]] = tensorrt.slice %arg1[0, 1][1, 1][1, 1] : tensor<1x2xi32> to tensor<1x1xi32>
//       CHECK:     %[[v1:.+]] = tensorrt.collapse_rank %[[v0]] : tensor<1x1xi32> to tensor<i32>
//       CHECK:     %[[v2:.+]] = tensorrt.collapse_rank %[[arg2]] : tensor<1x1x5xi1> to tensor<1x5xi1>
//       CHECK:     %[[cst_i32:.+]] = tensorrt.constant dense<1> : tensor<2xi32>
//       CHECK:     %[[v3:.+]] = tensorrt.linspace[%[[v1]] : tensor<i32>] [ static] [%[[cst_i32]] : tensor<2xi32>] : tensor<1x5xi32>
//       CHECK:     %[[v4:.+]] = tensorrt.identity %[[v2]] : tensor<1x5xi1> to tensor<1x5xi32>
//       CHECK:     %[[v5:.+]] = tensorrt.identity %[[arg0]] : tensor<1x134xi1> to tensor<1x134xi32>
//       CHECK:     %[[v6:.+]] = tensorrt.scatter_elements {axis = 1 : i64} data(%[[v5]] : tensor<1x134xi32>) indices(%[[v3]] : tensor<1x5xi32>) updates(%[[v4]] : tensor<1x5xi32>)
//       CHECK:     %[[v7:.+]] = tensorrt.identity %[[v6]] : tensor<1x134xi32> to tensor<1x134xi1>
//       CHECK:     return %[[v7]] : tensor<1x134xi1>

// -----

func.func @scatter_slice_update_i1_axis0(%arg0: tensor<1024x1xi1>, %arg1: tensor<1x1xi32>, %arg2: tensor<1x134x1xi1>) -> tensor<1024x1xi1> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%arg3: tensor<i1>, %arg4: tensor<i1>):
    stablehlo.return %arg4 : tensor<i1>
  }) : (tensor<1024x1xi1>, tensor<1x1xi32>, tensor<1x134x1xi1>) -> tensor<1024x1xi1>
  return %0 : tensor<1024x1xi1>
}

// CHECK-LABEL: @scatter_slice_update_i1_axis0
//       CHECK:     %[[v0:.+]] = tensorrt.slice %[[arg1]][0, 0][1, 1][1, 1] : tensor<1x1xi32> to tensor<1x1xi32>
//       CHECK:     %[[v1:.+]] = tensorrt.collapse_rank %[[v0]] : tensor<1x1xi32> to tensor<i32>
//       CHECK:     %[[v2:.+]] = tensorrt.collapse_rank %[[arg2]] : tensor<1x134x1xi1> to tensor<134x1xi1>
//       CHECK:     %[[cst_i32:.+]] = tensorrt.constant dense<1> : tensor<2xi32>
//       CHECK:     %[[v3:.+]] = tensorrt.linspace[%[[v1]] : tensor<i32>] [ static] [%[[cst_i32]] : tensor<2xi32>] : tensor<134x1xi32>
//       CHECK:     %[[v4:.+]] = tensorrt.identity %[[v2]] : tensor<134x1xi1> to tensor<134x1xi32>
//       CHECK:     %[[v5:.+]] = tensorrt.identity %[[arg0]] : tensor<1024x1xi1> to tensor<1024x1xi32>
//       CHECK:     %[[v6:.+]] = tensorrt.scatter_elements {axis = 0 : i64} data(%[[v5]] : tensor<1024x1xi32>) indices(%[[v3]] : tensor<134x1xi32>) updates(%[[v4]] : tensor<134x1xi32>)
//       CHECK:     %[[v7:.+]] = tensorrt.identity %[[v6]] : tensor<1024x1xi32> to tensor<1024x1xi1>
//       CHECK:     return %[[v7]] : tensor<1024x1xi1>

// -----

func.func @large_weight() -> tensor<258x256xf32> {
  %c = stablehlo.constant dense_resource<__elided__> : tensor<258x256xi4>
  %0 = stablehlo.composite "tensorrt.block_dq" %c {composite_attributes = {axis = -1 : i32, scale = dense_resource<__elided__> : tensor<2x256xf32>}, decomposition = @block_dq} : (tensor<258x256xi4>) -> tensor<258x256xf32>
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
  %0 = stablehlo.composite "tensorrt.pt_q" %cst {composite_attributes = {axis = -1 : i32, scale = dense<5.000000e-01> : tensor<bf16>}, decomposition = @pt_q} : (tensor<2xbf16>) -> tensor<2xf8E4M3FN>
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
