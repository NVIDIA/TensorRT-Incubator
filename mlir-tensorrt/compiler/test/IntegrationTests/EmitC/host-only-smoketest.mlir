// REQUIRES: system-linux
//
// RUN: rm -rf %t || true
// RUN: mkdir -p %t %t/build
// RUN: mlir-tensorrt-compiler %s -input=stablehlo -host-target=emitc -artifacts-dir=%t \
// RUN:   -disable-all-extensions -emitc-wrap-in-class -entrypoint= \
// RUN:   -emitc-emit-support-files -emitc-emit-cmake-file -o %t/host-only-smoketest.h
// RUN: %cmake -S %t -B %t/build
// RUN: %cmake --build %t/build
// RUN: env MTRT_ARTIFACTS_DIR=%t %t/build/emitc_test

!tensor_type = tensor<128xf32>

module @smoketest_tensorrt {

  func.func private @add(%arg0: !tensor_type, %arg1: !tensor_type)
      -> !tensor_type attributes {no_inline} {
    %0 = stablehlo.add %arg0, %arg1 : (!tensor_type, !tensor_type) -> !tensor_type
    return %0 : !tensor_type
  }

  func.func private @check_almost_eq(%arg0: !tensor_type, %arg1: !tensor_type) attributes {no_inline} {
    %0 = stablehlo.compare EQ, %arg0, %arg1 : (!tensor_type, !tensor_type) -> tensor<128xi1>
    %c0 = stablehlo.constant dense<1> : tensor<i1>
    %2 = stablehlo.reduce (%0 init: %c0)
      applies stablehlo.and across dimensions = [0]
       : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %3 = tensor.extract %2[] : tensor<i1>
    cf.assert %3, "check_almost_eq failed"
    return
  }

  func.func @main() {
    %lhs = stablehlo.constant dense<1.0> : tensor<128xf32>
    %rhs = stablehlo.constant dense<2.0> : tensor<128xf32>
    %expected = stablehlo.constant dense<3.0> : tensor<128xf32>
    %result = call @add(%lhs, %rhs) : (!tensor_type, !tensor_type) -> !tensor_type
    call @check_almost_eq(%result, %expected) : (!tensor_type, !tensor_type) -> ()
    return
  }
}
