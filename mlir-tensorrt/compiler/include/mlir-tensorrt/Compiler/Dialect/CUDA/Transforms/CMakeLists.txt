include_directories(${MLIR_INCLUDE_DIRS})

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name CUDA)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix CUDA)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix CUDA)
mtrt_add_public_tablegen_target(MLIRTensorRTCUDATransformsPassIncGen)

add_mlir_tensorrt_doc(CUDAPassesDoc
  SRC Passes.td
  OUTPUT_FILE docs/Passes/CUDAPasses.md
  COMMAND -gen-pass-doc
  )
