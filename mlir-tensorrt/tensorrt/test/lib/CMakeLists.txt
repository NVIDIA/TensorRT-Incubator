function(add_mlir_tensorrt_dialect_test_library target)
  add_mlir_library(${target} ${ARGN}
    EXCLUDE_FROM_LIBMLIR)
  set_property(GLOBAL APPEND PROPERTY MLIR_TENSORRT_DIALECT_TEST_LIBS ${target})
endfunction()

add_subdirectory(Target)
add_subdirectory(TensorRT)
