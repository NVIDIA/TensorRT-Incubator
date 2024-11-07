// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" -tensorrt-builder-opt-level=0 \
// RUN: --mlir-elide-elementsattrs-if-larger=32 %s | FileCheck %s

// CHECK-LABEL: @trt_fill_linspace
//  CHECK-SAME: tensorrt.engine
func.func @trt_fill_linspace() -> tensor<1024xf32> {
  %0 = tensorrt.linspace [0.0][static][1.0] : tensor<1024xf32>
  return %0 : tensor<1024xf32>
}

// CHECK-LABEL: @trt_fill_linspace_i32
//  CHECK-SAME: tensorrt.engine
func.func @trt_fill_linspace_i32() -> tensor<1024xi32> {
  %0 = tensorrt.linspace [0.0][static][1.0] : tensor<1024xi32>
  return %0 : tensor<1024xi32>
}

// CHECK-LABEL: @trt_fill_linspace_dynamic
//  CHECK-SAME: tensorrt.engine
func.func @trt_fill_linspace_dynamic() -> tensor<1024x1024xf32> {
  %shape = tensorrt.constant dense<[1024, 1024]>:tensor<2xi32>
  %start = tensorrt.constant dense<0.0>:tensor<f32>
  %step = tensorrt.constant dense<[1.0,1.0]>:tensor<2xf32>
  %0 = tensorrt.linspace [%start:tensor<f32>][%shape:tensor<2xi32>][%step:tensor<2xf32>] : tensor<1024x1024xf32>
  return %0 : tensor<1024x1024xf32>
}

func.func @trt_fill_linspace_dynamic_dim(%arg0: tensor<?x32xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [1, 32], opt = [64, 32], max = [64, 32]>}) -> tensor<?xf32> {
    %cst_i32 = tensorrt.constant dense<0> : tensor<1xi32>
    %0 = tensorrt.shape %arg0 : tensor<?x32xf32> -> tensor<2xi32>
    %1 = tensorrt.gather {axis = 0 : i64} ins(%0, %cst_i32 : tensor<2xi32>, tensor<1xi32>) -> tensor<1xi32>
    %2 = tensorrt.linspace[0.000000e+00] [%1 : tensor<1xi32>] [1.000000e+00] : tensor<?xi32>
    %3 = tensorrt.cast %2 : tensor<?xi32> to tensor<?xf32>
    return %3 : tensor<?xf32>
}
// CHECK-LABEL: @trt_fill_linspace_dynamic_dim
//  CHECK-SAME: tensorrt.engine