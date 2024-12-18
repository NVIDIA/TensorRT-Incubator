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

// CHECK-LABEL: @dynamic_nd_iota_1
//  CHECK-SAME: tensorrt.engine
func.func @dynamic_nd_iota_1(%arg0: tensor<2xi32> {
  tensorrt.value_bounds = #tensorrt.shape_profile<min=[1, 3], opt=[4, 3], max=[12, 3]>,
  tensorrt.host_tensor
}) -> tensor<?x3xi32> {
  %cst_i32 = tensorrt.constant dense<0> : tensor<i32>
  %cst_i32_0 = tensorrt.constant dense<[0, 1]> : tensor<2xi32>
  %0 = tensorrt.linspace[%cst_i32 : tensor<i32>] [%arg0 : tensor<2xi32>] [%cst_i32_0 : tensor<2xi32>] : tensor<?x3xi32>
  return %0 : tensor<?x3xi32>
}

// CHECK-LABEL: @dynamic_nd_iota_2
//  CHECK-SAME: tensorrt.engine
func.func @dynamic_nd_iota_2(%arg0: tensor<2xi32> {
  tensorrt.value_bounds = #tensorrt.shape_profile<min=[1, 3], opt=[4, 3], max=[12, 3]>,
  tensorrt.host_tensor
}) -> tensor<?x3xi32> {
  %cst_i32 = tensorrt.constant dense<0> : tensor<i32>
  %cst_i32_0 = tensorrt.constant dense<[1, 0]> : tensor<2xi32>
  %0 = tensorrt.linspace[%cst_i32 : tensor<i32>] [%arg0 : tensor<2xi32>] [%cst_i32_0 : tensor<2xi32>] : tensor<?x3xi32>
  return %0 : tensor<?x3xi32>
}
