function(add_mlir_kernel_test_library target)
  mtrt_add_test_library(${target}
    PROJECT_NAME Kernel
    IGNORE_LINK_MTRT
    ${ARGN}
    )
endfunction()

add_mlir_kernel_test_library(MLIRKernelBufferizationTestPass
  BufferizationTestPass.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    MLIRTensorRTKernelDialect
    MLIRTensorRTKernelTransforms
    MLIRTensorRTKernelPipelines
  MLIR_LIBS PUBLIC
    MLIRPass
    MLIRBufferizationDialect
  )

add_mlir_kernel_test_library(MLIRKernelTestOutlining
  TestOutlining.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    MLIRKernelOutliningUtils
    MLIRTensorRTKernelDialect
  MLIR_LIBS PUBLIC
    MLIRGPUDialect
  )

