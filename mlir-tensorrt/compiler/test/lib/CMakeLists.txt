function(add_mlir_tensorrt_test_library target)
  mtrt_add_test_library(${target} ${ARGN})
endfunction()

add_subdirectory(Analysis)
