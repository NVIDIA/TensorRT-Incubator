// REQUIRES: cuda
// REQUIRES: system-linux
//
// RUN: rm -rf %t || true
// RUN: mkdir -p %t/artifacts
// RUN: cp %S/cuda-launch-kernel.ptx $PWD/kernels.ptx
// RUN: mlir-tensorrt-opt %s -lower-affine -convert-host-to-emitc -cse -canonicalize -form-expressions \
// RUN:   -executor-serialize-artifacts="artifacts-directory=%t/artifacts create-manifest=true" \
// RUN:   --mlir-print-op-generic | \
// RUN: mlir-tensorrt-translate -mlir-to-cpp > %t/missing-artifacts.cpp
// RUN: %host_cxx \
// RUN:   %t/missing-artifacts.cpp \
// RUN:   %mtrt_src_dir/executor/lib/Runtime/StandaloneCPP/MTRTRuntimeStatus.cpp \
// RUN:   %mtrt_src_dir/executor/lib/Runtime/StandaloneCPP/MTRTRuntimeCore.cpp \
// RUN:   %mtrt_src_dir/executor/lib/Runtime/StandaloneCPP/MTRTRuntimeCuda.cpp \
// RUN:   %S/cuda-launch-argpack_driver.cpp \
// RUN:  -I%mtrt_src_dir/executor/lib/Runtime/StandaloneCPP \
// RUN:  %cuda_toolkit_linux_cxx_flags \
// RUN:  -o missing-artifacts-test
// RUN: env MTRT_ARTIFACTS_DIR=%t/does_not_exist \
// RUN:   not ./missing-artifacts-test 2>&1 | FileCheck %s --check-prefix=ERR


// ERR: Error opening file 'unnamed_module/kernels.ptx'.
// ERR: Tried:
// ERR:   - {{.*}}does_not_exist/unnamed_module/kernels.ptx
// ERR:   - unnamed_module/kernels.ptx
// ERR: Hint: set MTRT_ARTIFACTS_DIR to the directory containing manifest.json

cuda.compiled_module @kernels file "kernels.ptx"

func.func @run() -> i32 {
  %device = cuda.get_active_device
  %stream = cuda.get_global_stream device(%device)[0]
  %func = cuda.get_function "add_kernel" from @kernels
  %c0 = arith.constant 0 : i32
  return %c0 : i32
}
