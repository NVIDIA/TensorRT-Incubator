function(add_mlir_executor_test_library target)
  mtrt_add_test_library(${target}
    PROJECT_NAME MLIRExecutor
    IGNORE_LINK_MTRT
    ${ARGN}
    )
endfunction()

add_mlir_executor_test_library(MLIRExecutorBufferizationTestPass
  BufferizationTestPass.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTExecutorDialect
  MLIR_LIBS PUBLIC
    MLIRPass
    MLIRBufferizationTransforms
    MLIRBufferizationPipelines
    )

add_subdirectory(Transforms)
