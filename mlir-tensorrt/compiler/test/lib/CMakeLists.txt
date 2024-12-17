function(add_mlir_tensorrt_test_library target)
  add_mlir_tensorrt_library(${target} ${ARGN})
  set_property(GLOBAL APPEND PROPERTY MLIR_TENSORRT_TEST_LIBS ${target})
endfunction()

include_directories(${MLIR_TENSORRT_ROOT_DIR}/compiler/include)
include_directories(${MLIR_TENSORRT_ROOT_BINARY_DIR}/compiler/include)

add_subdirectory(Analysis)
add_subdirectory(Transforms)