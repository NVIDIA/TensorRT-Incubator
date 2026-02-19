// RUN: mlir-tensorrt-opt -split-input-file -convert-host-to-emitc %s | \
// RUN: mlir-tensorrt-translate -split-input-file -mlir-to-cpp | FileCheck %s --check-prefix=CPP

// Test arith.bitcast conversions to EmitC

func.func @bitcast_f32_to_i32(%arg0: f32) -> i32 {
  %0 = arith.bitcast %arg0 : f32 to i32
  return %0 : i32
}

// CPP-LABEL: int32_t bitcast_f32_to_i32(float v1) {
// CPP-NEXT:   int32_t v2 = mtrt::bit_cast<int32_t>(v1);
// CPP-NEXT:   return v2;
// CPP-NEXT: }

// -----

func.func @bitcast_i32_to_f32(%arg0: i32) -> f32 {
  %0 = arith.bitcast %arg0 : i32 to f32
  return %0 : f32
}

// CPP-LABEL: float bitcast_i32_to_f32(int32_t v1) {
// CPP-NEXT:   float v2 = mtrt::bit_cast<float>(v1);
// CPP-NEXT:   return v2;
// CPP-NEXT: }

// -----

func.func @bitcast_f64_to_i64(%arg0: f64) -> i64 {
  %0 = arith.bitcast %arg0 : f64 to i64
  return %0 : i64
}

// CPP-LABEL: int64_t bitcast_f64_to_i64(double v1) {
// CPP-NEXT:   int64_t v2 = mtrt::bit_cast<int64_t>(v1);
// CPP-NEXT:   return v2;
// CPP-NEXT: }

// -----

func.func @bitcast_i64_to_f64(%arg0: i64) -> f64 {
  %0 = arith.bitcast %arg0 : i64 to f64
  return %0 : f64
}

// CPP-LABEL: double bitcast_i64_to_f64(int64_t v1) {
// CPP-NEXT:   double v2 = mtrt::bit_cast<double>(v1);
// CPP-NEXT:   return v2;
// CPP-NEXT: }
