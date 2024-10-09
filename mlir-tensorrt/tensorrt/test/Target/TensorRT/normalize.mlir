// REQUIRES: long_tests
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" -tensorrt-builder-opt-level=0 \
// RUN: --mlir-elide-elementsattrs-if-larger=32  %s | FileCheck %s

func.func @trt_batch_normalize(%inp: tensor<2x3x2x2xf16>, %scale: tensor<1x3x1x1xf16>, %bias: tensor<1x3x1x1xf16>) -> tensor<2x3x2x2xf16> {
    %0 = tensorrt.normalization {
        axis = array<i64: 0>
    } (%inp: tensor<2x3x2x2xf16>, %scale: tensor<1x3x1x1xf16>, %bias: tensor<1x3x1x1xf16>) -> tensor<2x3x2x2xf16>
    return %0: tensor<2x3x2x2xf16>
}

// CHECK-LABEL: @trt_batch_normalize
//  CHECK-SAME: tensorrt.engine

// -----

func.func @trt_group_normalize(%inp: tensor<2x24x2x2xf16>, %scale: tensor<1x4x1x1xf16>, %bias: tensor<1x4x1x1xf16>) -> tensor<2x24x2x2xf16> {
    %0 = tensorrt.normalization {
        axis = array<i64: 2, 3>, num_groups = 4 : i32
    } (%inp: tensor<2x24x2x2xf16>, %scale: tensor<1x4x1x1xf16>, %bias: tensor<1x4x1x1xf16>) -> tensor<2x24x2x2xf16>
    return %0: tensor<2x24x2x2xf16>
}

// CHECK-LABEL: @trt_group_normalize
//  CHECK-SAME: tensorrt.engine

// -----

func.func @trt_instance_normalize(%inp: tensor<2x3x2x2xf16>, %scale: tensor<1x3x1x1xf16>, %bias: tensor<1x3x1x1xf16>) -> tensor<2x3x2x2xf16> {
    %0 = tensorrt.normalization {
        axis = array<i64: 2, 3>
    } (%inp: tensor<2x3x2x2xf16>, %scale: tensor<1x3x1x1xf16>, %bias: tensor<1x3x1x1xf16>) -> tensor<2x3x2x2xf16>
    return %0: tensor<2x3x2x2xf16>
}

// CHECK-LABEL: @trt_instance_normalize
//  CHECK-SAME: tensorrt.engine

// -----

func.func @trt_layer_normalize_last_2d(%inp: tensor<2x3x2x2x2xf16>, %scale: tensor<1x1x1x2x2xf16>, %bias: tensor<1x1x1x2x2xf16>) -> tensor<2x3x2x2x2xf16> {
    %0 = tensorrt.normalization {
        axis = array<i64: 3, 4>
    } (%inp: tensor<2x3x2x2x2xf16>, %scale: tensor<1x1x1x2x2xf16>, %bias: tensor<1x1x1x2x2xf16>) -> tensor<2x3x2x2x2xf16>
    return %0: tensor<2x3x2x2x2xf16>
}

// CHECK-LABEL: @trt_layer_normalize_last_2d
//  CHECK-SAME: tensorrt.engine

// -----

func.func @trt_layer_normalize_last_3d(%inp: tensor<2x3x2x2x2xf16>, %scale: tensor<1x1x2x2x2xf16>, %bias: tensor<1x1x2x2x2xf16>) -> tensor<2x3x2x2x2xf16> {
    %0 = tensorrt.normalization {
        axis = array<i64: 2, 3, 4>
    } (%inp: tensor<2x3x2x2x2xf16>, %scale: tensor<1x1x2x2x2xf16>, %bias: tensor<1x1x2x2x2xf16>) -> tensor<2x3x2x2x2xf16>
    return %0: tensor<2x3x2x2x2xf16>
}

// CHECK-LABEL: @trt_layer_normalize_last_3d
//  CHECK-SAME: tensorrt.engine
