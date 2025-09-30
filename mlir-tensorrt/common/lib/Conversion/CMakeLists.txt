set(LLVM_TARGET_DEFINITIONS
  "${MLIR_TENSORRT_COMMON_SOURCE_DIR}/include/mlir-tensorrt-common/Conversion/Passes.td")
set(OUTPUT_DIR
  "${MLIR_TENSORRT_COMMON_BINARY_DIR}/include/mlir-tensorrt-common/Conversion")

cmake_path(SET OUTPUT_DIR NORMALIZE "${OUTPUT_DIR}")
cmake_path(RELATIVE_PATH OUTPUT_DIR BASE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
mlir_tablegen("${OUTPUT_DIR}/Passes.h.inc" -gen-pass-decls -name MLIRTensorRTCommonConversion)
mtrt_add_public_tablegen_target(MLIRTensorRTCommonConversionPassesIncGen)

add_subdirectory(ToLoops)
