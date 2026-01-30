
// RUN: rm -rf %t || true
// RUN: mkdir %t
// RUN: %pick-one-gpu tensorrt-opt %s -pass-pipeline="builtin.module(tensorrt.module(translate-tensorrt-to-engine))" \
// RUN:  --mlir-elide-elementsattrs-if-larger=32 \
// RUN:  --tensorrt-save-engines-dir=%t
// RUN: %pick-one-gpu tensorrt-opt %s -pass-pipeline="builtin.module(trensorrt.module(translate-tensorrt-to-engine))" \
// RUN:  --mlir-elide-elementsattrs-if-larger=32 \
// RUN:  --tensorrt-load-engines-dir=%t


tensorrt.module @sub_module {
  func.func @func1(%arg0: tensor<2x10xf32>) -> tensor<2x10xf32> {
    return %arg0 : tensor<2x10xf32>
  }
  func.func @func2(%arg0: tensor<2x10xf32>) -> tensor<2x10xf32> {
    return %arg0: tensor<2x10xf32>
  }
}
