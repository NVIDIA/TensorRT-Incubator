add_mlir_tensorrt_dialect_library(MLIRTensorRTTensorRTRuntimeDialect
  TensorRTRuntime.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_TENSORRT_ROOT_DIR}/include

  DEPENDS
  MLIRTensorRTRuntimeOpsIncGen

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRDialect
  MLIRExecutorCommonUtils
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSupport
  MLIRTensorRTCUDADialect
  MLIRTensorRTTensorRTDialect
  )
