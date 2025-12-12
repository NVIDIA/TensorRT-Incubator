// RUN: kernel-opt --kernel-normalize-quantized-conversions -split-input-file %s | FileCheck %s

gpu.module @normalize_quantized_conversions{
  func.func @bf16_to_fp8(%arg0: bf16) -> f8E4M3FN {
    %0 = arith.truncf %arg0 : bf16 to f8E4M3FN
    return %0 : f8E4M3FN
  }
}

// CHECK-LABEL: @bf16_to_fp8
// CHECK: %[[F32:.*]] = arith.extf %arg0 : bf16 to f32
// CHECK: %[[F16:.*]] = arith.truncf %[[F32]] : f32 to f16
// CHECK: %[[FP8:.*]] = arith.truncf %[[F16]] : f16 to f8E4M3FN
// CHECK: return %[[FP8]]

// -----

gpu.module @normalize_quantized_conversions{
  func.func @bf16_to_fp4(%arg0: bf16) -> f4E2M1FN {
    %0 = arith.truncf %arg0 : bf16 to f4E2M1FN
    return %0 : f4E2M1FN
  }
}

// CHECK-LABEL: @bf16_to_fp4
// CHECK: %[[F32:.*]] = arith.extf %arg0 : bf16 to f32
// CHECK: %[[F16:.*]] = arith.truncf %[[F32]] : f32 to f16
// CHECK: %[[FP4:.*]] = arith.truncf %[[F16]] : f16 to f4E2M1FN
// CHECK: return %[[FP4]]

// -----

gpu.module @normalize_quantized_conversions{
  func.func @fp8_to_bf16(%arg0: f8E4M3FN) -> bf16 {
    %0 = arith.extf %arg0 : f8E4M3FN to bf16
    return %0 : bf16
  }
}

// CHECK-LABEL: @fp8_to_bf16
// CHECK: %[[F16:.*]] = arith.extf %arg0 : f8E4M3FN to f16
// CHECK: %[[F32:.*]] = arith.extf %[[F16]] : f16 to f32
// CHECK: %[[BF16:.*]] = arith.truncf %[[F32]] : f32 to bf16
// CHECK: return %[[BF16]]

// -----

gpu.module @normalize_quantized_conversions{
  func.func @fp4_to_bf16(%arg0: f4E2M1FN) -> bf16 {
    %0 = arith.extf %arg0 : f4E2M1FN to bf16
    return %0 : bf16
  }
}

// CHECK-LABEL: @fp4_to_bf16
// CHECK: %[[F16:.*]] = arith.extf %arg0 : f4E2M1FN to f16
// CHECK: %[[F32:.*]] = arith.extf %[[F16]] : f16 to f32
// CHECK: %[[BF16:.*]] = arith.truncf %[[F32]] : f32 to bf16
// CHECK: return %[[BF16]]

// -----

gpu.module @normalize_quantized_conversions{
  func.func @f16_to_fp8_unchanged(%arg0: f16) -> f8E4M3FN {
    %0 = arith.truncf %arg0 : f16 to f8E4M3FN
    return %0 : f8E4M3FN
  }
}
// CHECK-LABEL: @f16_to_fp8_unchanged
//  CHECK-NEXT: %[[FP8:.*]] = arith.truncf %arg0 : f16 to f8E4M3FN
//  CHECK-NEXT: return %[[FP8]]

// -----

gpu.module @normalize_quantized_conversions{
  func.func @fp8_to_f16_unchanged(%arg0: f8E4M3FN) -> f16 {
    %0 = arith.extf %arg0 : f8E4M3FN to f16
    return %0 : f16
  }
}

// CHECK-LABEL: @fp8_to_f16_unchanged
//  CHECK-NEXT: %[[F16:.*]] = arith.extf %arg0 : f8E4M3FN to f16
//  CHECK-NEXT: return %[[F16]]

// -----

gpu.module @normalize_quantized_conversions{
  func.func @f32_to_fp8(%arg0: f32) -> f8E4M3FN {
    %0 = arith.truncf %arg0 : f32 to f8E4M3FN
    return %0 : f8E4M3FN
  }
}

// CHECK-LABEL: @f32_to_fp8
// CHECK: %[[FP16:.*]] = arith.truncf %arg0 : f32 to f16
// CHECK: %[[FP8:.*]] = arith.truncf %[[FP16]] : f16 to f8E4M3FN
// CHECK: return %[[FP8]]