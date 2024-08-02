# LLVM `mlir_tablegen` function depends grabbing include directories
# from the scope, so we must use `include_directories` here.
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${MLIR_TENSORRT_ROOT_DIR}/include)

add_mlir_dialect(TensorRTRuntimeOps trtrt)
add_mlir_tensorrt_doc(TensorRTRuntimeDialect
  SRC TensorRTRuntimeOps.td
  OUTPUT_FILE docs/Dialects/TensorRTRuntimeDialect.md
  COMMAND -gen-dialect-doc -dialect=trtrt
  )
