# LLVM `mlir_tablegen` function depends grabbing include directories
# from the scope, so we must use `include_directories` here.
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${MLIR_TENSORRT_ROOT_DIR}/include)

add_mlir_dialect(CUDAOps cuda)
add_mlir_tensorrt_doc(CUDADialect
  SRC CUDAOps.td
  OUTPUT_FILE docs/Dialects/CUDADialect.md
  COMMAND -gen-dialect-doc -dialect=cuda
  )

# Op Interfaces
add_mlir_interface(CUDAInterfaces)
