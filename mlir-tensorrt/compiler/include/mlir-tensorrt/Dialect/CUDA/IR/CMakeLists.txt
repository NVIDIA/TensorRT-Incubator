# LLVM `mlir_tablegen` function depends grabbing include directories
# from the scope, so we must use `include_directories` here.
add_mlir_dialect(CUDAOps cuda)

set(LLVM_TARGET_DEFINITIONS CUDAEnums.td)
mlir_tablegen(CUDAEnums.h.inc -gen-enum-decls)
mlir_tablegen(CUDAEnums.cpp.inc -gen-enum-defs)
mtrt_add_public_tablegen_target(MLIRCUDAEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS CUDAAttributes.td)
mlir_tablegen(CUDAAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(CUDAAttributes.cpp.inc -gen-attrdef-defs)
mtrt_add_public_tablegen_target(MLIRCUDAAttributesIncGen)

add_mlir_tensorrt_doc(CUDADialect
  SRC CUDAOps.td
  OUTPUT_FILE docs/Dialects/CUDADialect.md
  COMMAND -gen-dialect-doc -dialect=cuda
  )

# Op Interfaces
add_mlir_interface(CUDAInterfaces)
