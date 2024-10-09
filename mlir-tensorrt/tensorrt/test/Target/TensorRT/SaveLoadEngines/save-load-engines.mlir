
// RUN: rm -rf %t || true
// RUN: mkdir %t
// RUN: %pick-one-gpu tensorrt-opt %s -pass-pipeline="builtin.module(tensorrt.module(translate-tensorrt-to-engine))" \
// RUN:  --mlir-elide-elementsattrs-if-larger=32 \
// RUN:  --save-tensorrt-engines=%t
// RUN: %pick-one-gpu tensorrt-opt %s -pass-pipeline="builtin.module(trensorrt.module(translate-tensorrt-to-engine))" \
// RUN:  --mlir-elide-elementsattrs-if-larger=32 \
// RUN:  --load-tensorrt-engines=%t


tensorrt.module @sub_module {
  func.func @func1(%arg0: tensor<2x10xf32>) -> tensor<2x10xf32> {
    return %arg0 : tensor<2x10xf32>
  }
  func.func @func2(%arg0: tensor<2x10xf32>) -> tensor<2x10xf32> {
    return %arg0: tensor<2x10xf32>
  }
}
