function(add_mlir_executor_test_library target)
  mtrt_add_project_library(${target}
    PROJECT_NAME MLIRExecutor
    EXCLUDE_FROM_LIBMLIR
    DISABLE_INSTALL
    LIBRARY_TYPE TEST_LIBS
    PARTIAL_SOURCES_INTENDED
    ${ARGN}
    )
endfunction()

add_mlir_executor_test_library(MLIRExecutorBufferizationTestPass
  BufferizationTestPass.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTensorRTExecutorDialect
  MLIRBufferizationTransforms
  MLIRBufferizationPipelines)

add_subdirectory(Transforms)
