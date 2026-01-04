// REQUIRES: cuda
// REQUIRES: tensorrt
// REQUIRES: host-has-at-least-1-gpus
// DEFINE: %{prefix} = mlir-tensorrt-compiler --mlir-elide-elementsattrs-if-larger=16 --mlir-elide-resource-strings-if-larger=16 -artifacts-dir=%t -o -

// RUN: rm -rf %t || true
// RUN: %{prefix} %s \
// RUN:   -disable-all-extensions \
// RUN:   -host-target=executor -mlir | FileCheck %s --check-prefix=HOST_EXEC_MLIR

// RUN: rm -rf %t || true
// RUN: %{prefix} %s \
// RUN:   --disable-all-extensions \
// RUN:   --host-target=llvm -mlir | FileCheck %s --check-prefix=HOST_LLVM_MLIR
// RUN: not test -f %t/manifest.json

// RUN: rm -rf %t || true
// RUN: %{prefix} %s \
// RUN:   --disable-kernel-gen-extension \
// RUN:   --host-target=llvm -mlir | FileCheck %s --check-prefix=TRT_ONLY_LLVM_MLIR
// RUN: test -f %t/manifest.json
// RUN: test -f %t/smoketest/tensorrt_cluster_engine_data.trt_plan.bin

// RUN: rm -rf %t || true
// RUN: %{prefix} %s \
// RUN:   --disable-all-extensions \
// RUN:   --host-target=emitc -mlir | FileCheck %s --check-prefix=HOST_EMITC_MLIR
// RUN: not test -f %t/manifest.json

// RUN: rm -rf %t || true
// RUN: %{prefix} %s \
// RUN:   --disable-all-extensions \
// RUN:   --host-target=emitc | FileCheck %s --check-prefix=HOST_EMITC_CPP
// RUN: not test -f %t/manifest.json

// RUN: rm -rf %t || true
// RUN: %{prefix} %s \
// RUN:   --disable-kernel-gen-extension \
// RUN:   --host-target=emitc | FileCheck %s --check-prefix=TRT_ONLY_EMITC_CPP
// RUN: test -f %t/manifest.json
// RUN: test -f %t/smoketest/tensorrt_cluster_engine_data.trt_plan.bin

// RUN: rm -rf %t || true
// RUN: %{prefix} %s \
// RUN:   --disable-tensorrt-extension \
// RUN:   --host-target=emitc | FileCheck %s --check-prefix=KGEN_ONLY_EMITC_CPP
// RUN: test -f %t/manifest.json
// RUN: test -f %t/smoketest/codegen_cluster_kernel_cuModule_0.ptx


module @smoketest {
  func.func @main(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<128xf32>
    return %0 : tensor<128xf32>
  }
}

// HOST_EXEC_MLIR: module @smoketest
// HOST_EXEC_MLIR: func.func {{(public )?}}@main({{.*}})

// HOST_LLVM_MLIR: module @smoketest
// HOST_LLVM_MLIR: llvm.func @main({{.*}})
// HOST_LLVM_MLIR: llvm.func @_mlir_ciface_main({{.*}})

// TRT_ONLY_LLVM_MLIR: module @smoketest
// TRT_ONLY_LLVM_MLIR: llvm.func @main({{.*}})
// TRT_ONLY_LLVM_MLIR: llvm.func @_mlir_ciface_main({{.*}})

// HOST_EMITC_MLIR: module @smoketest
// HOST_EMITC_MLIR: func.func {{(public )?}}@main({{.*}})

// HOST_EMITC_CPP-NOT: MTRTRuntimeCuda.h
// HOST_EMITC_CPP: void main(mtrt::RankedMemRef<1>* v1, mtrt::RankedMemRef<1>* v2, mtrt::RankedMemRef<1>* v3)
// HOST_EMITC_CPP-NOT: cuda

// TRT_ONLY_EMITC_CPP: #include "MTRTRuntimeTensorRT.h"
// TRT_ONLY_EMITC_CPP: void main(mtrt::RankedMemRef<1>* v1, mtrt::RankedMemRef<1>* v2, mtrt::RankedMemRef<1>* v3)
// TRT_ONLY_EMITC_CPP: int32_t smoketest_initialize_all()
// TRT_ONLY_EMITC_CPP: void smoketest_destroy_all()

// KGEN_ONLY_EMITC_CPP-NOT: #include "MTRTRuntimeTensorRT.h"
// KGEN_ONLY_EMITC_CPP: void main(mtrt::RankedMemRef<1>* v1, mtrt::RankedMemRef<1>* v2, mtrt::RankedMemRef<1>* v3)
// KGEN_ONLY_EMITC_CPP: int32_t smoketest_initialize_all()
// KGEN_ONLY_EMITC_CPP:   int32_t {{.*}} = smoketest_{{.*cuModule.*}}_initialize();
// KGEN_ONLY_EMITC_CPP: void smoketest_destroy_all()
// KGEN_ONLY_EMITC_CPP:   smoketest_codegen_{{.*cuModule.*}}_destroy({{.*}});
