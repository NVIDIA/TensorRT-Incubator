// RUN: mlir-tensorrt-compiler --print-pass-pipeline=pretty \
// RUN:   --disable-tensorrt-extension --disable-kernel-gen-extension \
// RUN:   --host-target=executor %s 2>&1 | FileCheck %s --check-prefix=EXEC

// RUN: mlir-tensorrt-compiler --print-pass-pipeline=pretty \
// RUN:   --disable-tensorrt-extension --disable-kernel-gen-extension \
// RUN:   --host-target=llvm %s 2>&1 | FileCheck %s --check-prefix=LLVM

// RUN: mlir-tensorrt-compiler --print-pass-pipeline=pretty \
// RUN:   --disable-tensorrt-extension --disable-kernel-gen-extension \
// RUN:   --host-target=emitc %s 2>&1 | FileCheck %s --check-prefix=EMITC
//
// RUN: mlir-tensorrt-compiler --print-pass-pipeline=pretty \
// RUN:   --disable-tensorrt-extension --disable-kernel-gen-extension \
// RUN:   --host-target=executor --abi-version=0 %s 2>&1 | FileCheck %s --check-prefix=ABI0

// RUN: mlir-tensorrt-compiler --print-pass-pipeline=pretty \
// RUN:   --disable-tensorrt-extension --disable-kernel-gen-extension \
// RUN:   --host-target=executor --abi-version=1 %s 2>&1 | FileCheck %s --check-prefix=ABI1
//
// RUN: mlir-tensorrt-compiler --print-pass-pipeline=pretty \
// RUN:   --input=tensorrt --host-target=llvm %s 2>&1 | FileCheck %s --check-prefix=TRT_LLVM

builtin.module {
  func.func @main() {
    return
  }
}

// EXEC: Loaded Extensions:
// EXEC: Pass Pipeline:
// Anchor a few Plan setup passes.
// EXEC: plan-populate-default-backend-metadata
// EXEC: plan-legalize-io-bounds-attributes
// EXEC: plan-verify-input-and-assign-slots
// Executor ABI wrapper generation is enabled for abi>=1 when host-target!=llvm.
// EXEC: executor-generate-abi-wrappers{force-undef-output-args=false}
// EXEC: plan-create-closed-regions{input=stablehlo prefer-alloc-calling-convention=false
// EXEC: plan-alloc-tensors{force-entrypoints-return-allocs=false}
// Executor path should include lowering to executor.
// EXEC: convert-plan-to-executor

// LLVM: Loaded Extensions:
// LLVM: Pass Pipeline:
// LLVM: plan-verify-input-and-assign-slots
// LLVM: convert-scf-to-cf
// LLVM-NOT: convert-plan-to-executor
// LLVM-NOT: executor-generate-abi-wrappers

// EMITC: Loaded Extensions:
// EMITC: Pass Pipeline:
// EmitC lowering preserves SCF for readability.
// EMITC-NOT: convert-scf-to-cf
// EMITC-NOT: convert-plan-to-executor
// ABI wrapper generation is enabled for abi>=1 when host-target!=llvm (includes emitc).
// EMITC: executor-generate-abi-wrappers

// ABI0: Pass Pipeline:
// ABI0-NOT: executor-generate-abi-wrappers
// ABI0: convert-plan-to-executor
// ABI0: executor-populate-func-metadata

// ABI1: Pass Pipeline:
// ABI1: executor-generate-abi-wrappers
// ABI1: convert-plan-to-executor
// ABI1-NOT: executor-populate-func-metadata

// TRT_LLVM: Pass Pipeline:
// TRT_LLVM-NOT: executor-generate-abi-wrappers
// TRT_LLVM: plan-alloc-tensors{force-entrypoints-return-allocs=true}
// TRT_LLVM: convert-tensorrt-runtime-to-llvm
// TRT_LLVM: convert-cuda-to-llvm
// TRT_LLVM: convert-host-to-llvm
