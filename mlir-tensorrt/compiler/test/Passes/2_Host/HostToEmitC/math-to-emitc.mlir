// RUN: mlir-tensorrt-opt -split-input-file -convert-host-to-emitc %s | \
// RUN: mlir-tensorrt-translate -split-input-file -mlir-to-cpp | FileCheck %s --check-prefix=CPP

// Test Math Log conversion to EmitC

func.func @math_log_f32(%arg0: f32) -> f32 {
  %0 = math.log %arg0 : f32
  return %0 : f32
}

// CPP-LABEL: float math_log_f32(float v1) {
// CPP-NEXT:   float v2 = logf(v1);
// CPP-NEXT:   return v2;
// CPP-NEXT: }

// -----

func.func @math_log_f64(%arg0: f64) -> f64 {
  %0 = math.log %arg0 : f64
  return %0 : f64
}

// CPP-LABEL: double math_log_f64(double v1) {
// CPP-NEXT:   double v2 = log(v1);
// CPP-NEXT:   return v2;
// CPP-NEXT: }
