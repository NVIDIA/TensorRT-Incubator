// RUN: mlir-tensorrt-opt -split-input-file %s --convert-stablehlo-to-tensorrt -verify-diagnostics | FileCheck %s

func.func @stablehlo_all_reduce_region(%arg0 : tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.all_reduce"(%arg0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }) {channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<0> : tensor<1x1xi64>} : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}


// CHECK-LABEL: @stablehlo_all_reduce_region
//       CHECK: stablehlo.all_reduce
//   CHECK-NOT:   tensorrt
//       CHECK:   stablehlo.add

// -----

func.func @uniform_quantize_zero_point_unsupported(%arg: tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<i8:f32, 34.0:16>> {
  %0 = "stablehlo.uniform_quantize"(%arg) : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<i8:f32, 34.0:16>>
  return %0 : tensor<16x16x!quant.uniform<i8:f32, 34.0:16>>
}

// CHECK-LABEL: @uniform_quantize_zero_point_unsupported
//   CHECK-NOT: tensorrt
//       CHECK: stablehlo.uniform_quantize

// -----

func.func @uniform_dequantize_zero_point_unsupported(%arg: tensor<16x16x!quant.uniform<i8:f32, 34.0:16>>) -> tensor<16x16xf32> {
  %0 = "stablehlo.uniform_dequantize"(%arg) : (tensor<16x16x!quant.uniform<i8:f32, 34.0:16>>) -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

// CHECK-LABEL: @uniform_dequantize_zero_point_unsupported
//   CHECK-NOT: tensorrt
//       CHECK: stablehlo.uniform_dequantize


// -----

func.func @not_i32_unsupported(%arg0: tensor<128xi32>) -> tensor<128xi32> {
  %0 = stablehlo.not %arg0 : tensor<128xi32>
  return %0 : tensor<128xi32>
}

// CHECK-LABEL: @not_i32_unsupported
//   CHECK-NOT: tensorrt

// -----

// expected-error @below {{failed to legalize operation 'func.func' that was explicitly marked illegal}}
func.func @i16_convert_unsupported(%arg0: tensor<150528xi32>) -> tensor<150528xi16> {
  %0 = stablehlo.convert %arg0 : (tensor<150528xi32>) -> tensor<150528xi16>
  return %0 : tensor<150528xi16>
}

// -----

func.func @hlo_sort_invalid() -> tensor<4000xi32> {
    %cst = stablehlo.constant dense<3> : tensor<4000xi32>
    %1 = "stablehlo.sort"(%cst) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %2 = stablehlo.compare  LT, %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    }) {dimension = 0 : i64, is_stable = false} : (tensor<4000xi32>) -> tensor<4000xi32>
    return %1 : tensor<4000xi32>
}

// CHECK-LABEL: @hlo_sort_invalid
//       CHECK: tensorrt.constant
//       CHECK: stablehlo.sort
//       CHECK: stablehlo.compare
//       CHECK: stablehlo.return
//       CHECK: return

// -----

func.func @large_weight() -> tensor<258x256xf32> {
  %c = stablehlo.constant dense_resource<__elided__> : tensor<258x256xi4>
  // expected-error @below {{failed to legalize operation 'stablehlo.composite' that was explicitly marked illegal}}
  %0 = stablehlo.composite "tensorrt.block_dq" %c {composite_attributes = {is_pointwise, scale = dense_resource<__elided__> : tensor<2x256xf32>}, decomposition = @block_dq} : (tensor<258x256xi4>) -> tensor<258x256xf32>
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