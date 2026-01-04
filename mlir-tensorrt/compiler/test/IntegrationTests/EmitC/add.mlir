// REQUIRES: host-has-at-least-1-gpus
// REQUIRES: cuda
// REQUIRES: system-linux
//
// RUN: rm -rf %t || true
// RUN: mkdir -p %t
// RUN: mlir-tensorrt-compiler %s -input=stablehlo -host-target=emitc -o %t/add.h -artifacts-dir=%t \
// RUN:   -disable-kernel-gen-extension -emitc-wrap-in-class -entrypoint=
// RUN: %host_cxx \
// RUN:   %S/add_driver.cpp \
// RUN:   %mtrt_src_dir/executor/lib/Runtime/StandaloneCPP/MTRTRuntimeStatus.cpp \
// RUN:   %mtrt_src_dir/executor/lib/Runtime/StandaloneCPP/MTRTRuntimeCore.cpp \
// RUN:   %mtrt_src_dir/executor/lib/Runtime/StandaloneCPP/MTRTRuntimeCuda.cpp \
// RUN:   %mtrt_src_dir/executor/lib/Runtime/StandaloneCPP/MTRTRuntimeTensorRT.cpp \
// RUN:  -I%mtrt_src_dir/executor/lib/Runtime/StandaloneCPP \
// RUN:  -I%t \
// RUN:  %cuda_toolkit_linux_cxx_flags \
// RUN:  -I%nvinfer_include_dir \
// RUN:  -L%nvinfer_lib_dir \
// RUN:  -lnvinfer \
// RUN:  -o add-test
// RUN: env MTRT_ARTIFACTS_DIR=%t ./add-test

!tensor_type = tensor<128xf32>

module @smoketest_tensorrt {

  func.func private @add(%arg0: !tensor_type, %arg1: !tensor_type)
      -> !tensor_type attributes {no_inline} {
    %0 = tensorrt.element_wise <kSUM> (%arg0, %arg1 : !tensor_type, !tensor_type) -> !tensor_type
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
