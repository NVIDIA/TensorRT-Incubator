add_mlir_tensorrt_dialect_library(MLIRTensorRTTensorRTRuntimeDialect
  TensorRTRuntime.cpp

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
