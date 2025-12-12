# LLVM `mlir_tablegen` function depends grabbing include directories
# from the scope, so we must use `include_directories` here.
include_directories(${MLIR_INCLUDE_DIRS})

set(LLVM_TARGET_DEFINITIONS KernelTransformOps.td)
mlir_tablegen(KernelTransformOps.h.inc -gen-op-decls)
mlir_tablegen(KernelTransformOps.cpp.inc -gen-op-defs)
mtrt_add_public_tablegen_target(MLIRTensorRTKernelTransformOpsIncGen)
