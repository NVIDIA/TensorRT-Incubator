function(add_mlir_executor_test_library target)
  add_mlir_library(${target} ${ARGN}
    EXCLUDE_FROM_LIBMLIR)
  set_property(GLOBAL APPEND PROPERTY MLIR_EXECUTOR_TEST_LIBS ${target})
endfunction()

add_mlir_executor_test_library(MLIRExecutorBufferizationTestPass
  BufferizationTestPass.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTensorRTExecutorDialect
  MLIRBufferizationTransforms
  MLIRBufferizationPipelines)

add_subdirectory(Transforms)
