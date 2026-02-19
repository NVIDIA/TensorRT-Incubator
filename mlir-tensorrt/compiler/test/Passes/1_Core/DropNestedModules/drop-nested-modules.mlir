// RUN: mlir-tensorrt-opt %s --split-input-file --drop-nested-modules | FileCheck %s

module @some_module {
  func.func @nested_func() {
    return
  }
  module @some_nested_module {
    func.func @nested_nested_func() {
      return
    }
  }
}

// CHECK-LABEL: @some_module
//  CHECK-NEXT:   @nested_func
//   CHECK-NOT:   @some_nested_module
