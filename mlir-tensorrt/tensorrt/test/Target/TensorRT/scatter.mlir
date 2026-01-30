// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" -tensorrt-builder-opt-level=0 \
// RUN: --mlir-elide-elementsattrs-if-larger=32  %s | FileCheck %s

func.func @trt_scatter_default() -> tensor<4x4x4xf32> {
  %data = tensorrt.constant dense<[
    [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [8.0, 7.0, 6.0, 5.0], [4.0, 3.0, 2.0, 1.0]],
    [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [8.0, 7.0, 6.0, 5.0], [4.0, 3.0, 2.0, 1.0]],
    [[8.0, 7.0, 6.0, 5.0], [4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
    [[8.0, 7.0, 6.0, 5.0], [4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    ]> : tensor<4x4x4xf32>
  %indices = tensorrt.constant dense<[[2], [1]]> : tensor<2x1xi32>
  %updates = tensorrt.constant dense<[
    [[5.0, 5.0, 5.0, 5.0],[6.0, 6.0, 6.0, 6.0], [7.0, 7.0, 7.0, 7.0], [8.0, 8.0, 8.0, 8.0]],
    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0]]
    ]> : tensor<2x4x4xf32>

  %result = tensorrt.scatter_nd
    data(%data: tensor<4x4x4xf32>)
    indices(%indices: tensor<2x1xi32>)
    updates(%updates: tensor<2x4x4xf32>)

  return %result: tensor<4x4x4xf32>
}

// CHECK-LABEL: @trt_scatter_default
//  CHECK-SAME: tensorrt.engine

// Should produce [I] 1 2 3 4 5 6 7 8 8 7 6 5 4 3 2 1
//                    1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4
//                    5 5 5 5 6 6 6 6 7 7 7 7 8 8 8 8
//                    8 7 6 5 4 3 2 1 1 2 3 4 5 6 7 8


func.func @trt_scatter_nd_f16(%data: tensor<4x4x4xf16>, %updates: tensor<2x4x4xf16>) -> tensor<4x4x4xf16> {
  %indices = tensorrt.constant dense<[[2], [1]]> : tensor<2x1xi32>
  %result = tensorrt.scatter_nd
    data(%data: tensor<4x4x4xf16>)
    indices(%indices: tensor<2x1xi32>)
    updates(%updates: tensor<2x4x4xf16>)
  return %result: tensor<4x4x4xf16>
}

// CHECK-LABEL: @trt_scatter_nd_f16
//  CHECK-SAME: tensorrt.engine


// CHECK-LABEL: @trt_scatter_1d_index
//  CHECK-SAME: tensorrt.engine

// A 1-d index implies that the update is a single 0-d scalar.
func.func @trt_scatter_1d_index() -> tensor<2x2xf32> {
  %data = tensorrt.constant dense<
    [[1.0, 2.0],
     [3.0, 4.0]] > : tensor<2x2xf32>
  %indices = tensorrt.constant dense<[0, 0]> : tensor<2xi32>
  %updates = tensorrt.constant dense<0.0> : tensor<f32>
  %result = tensorrt.scatter_nd
    data(%data: tensor<2x2xf32>)
    indices(%indices: tensor<2xi32>)
    updates(%updates: tensor<f32>)
  return %result: tensor<2x2xf32>
}


//-----

// CHECK-LABEL: @trt_scatter_elements
//  CHECK-SAME: tensorrt.engine

func.func @trt_scatter_elements() -> tensor<3x3xf32> {
  %data = tensorrt.constant dense<0.0> : tensor<3x3xf32>
  %indices = tensorrt.constant dense<[
    [1, 0, 2],
    [0, 2, 1]]> : tensor<2x3xi32>
  %updates = tensorrt.constant dense<[
    [1.0, 1.1, 1.2],
    [2.0, 2.1, 2.2]
    ]> : tensor<2x3xf32>

  %result = tensorrt.scatter_elements
    data(%data: tensor<3x3xf32>)
    indices(%indices: tensor<2x3xi32>)
    updates(%updates: tensor<2x3xf32>)

  return %result: tensor<3x3xf32>
}

// Should produce output
// [2.0, 1.1, 0.0]
// [1.0, 0.0, 2.2]
// [0.0, 2.1, 1.2]

//-----

// CHECK-LABEL: @trt_scatter_elements
//  CHECK-SAME: tensorrt.engine

func.func @trt_scatter_elements_2() -> tensor<1x5xf32> {
  %data = tensorrt.constant dense<[[1.0, 2.0, 3.0, 4.0, 5.0]]> : tensor<1x5xf32>
  %indices = tensorrt.constant dense<[[1, 3]]> : tensor<1x2xi32>
  %updates = tensorrt.constant dense<[[1.1, 2.1]]> : tensor<1x2xf32>

  %result = tensorrt.scatter_elements
    {axis = 1 : i64}
    data(%data: tensor<1x5xf32>)
    indices(%indices: tensor<1x2xi32>)
    updates(%updates: tensor<1x2xf32>)

  return %result: tensor<1x5xf32>
}

// should produce output
// [[1.0, 1.1, 3.0, 2.1, 5.0]]
