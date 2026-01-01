// REQUIRES: cuda
// REQUIRES: tensorrt
// REQUIRES: host-has-at-least-1-gpus
// DEFINE: %{prefix} = mlir-tensorrt-compiler --mlir-elide-elementsattrs-if-larger=16 --mlir-elide-resource-strings-if-larger=16 -artifacts-dir=%t -o -

// RUN: rm -rf %t || true
// RUN: not %{prefix} %s \
// RUN:   -disable-all-extensions \
// RUN:   -input=tensorrt \
// RUN:   -host-target=executor -mlir

// RUN: rm -rf %t || true
// RUN: %{prefix} %s \
// RUN:   -input=tensorrt \
// RUN:   -host-target=executor -mlir | FileCheck %s --check-prefix=EXEC_MLIR

// RUN: rm -rf %t || true
// RUN: %{prefix} %s \
// RUN:   --host-target=llvm -mlir | FileCheck %s --check-prefix=LLVM_MLIR
// RUN: test -f %t/manifest.json

// RUN: rm -rf %t || true
// RUN: %{prefix} %s \
// RUN:   --host-target=emitc | FileCheck %s --check-prefix=EMITC_CPP
// RUN: test -f %t/manifest.json

module @smoketest_tensorrt {
  func.func @main(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %0 = tensorrt.element_wise <kSUM> (%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }
}

// EXEC_MLIR: module @smoketest_tensorrt

// LLVM_MLIR: module @smoketest_tensorrt

// EMITC_CPP: void main
