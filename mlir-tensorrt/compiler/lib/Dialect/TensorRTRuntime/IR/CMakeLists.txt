add_mlir_tensorrt_dialect_library(MLIRTensorRTTensorRTRuntimeDialect
  TensorRTRuntime.cpp

  DEPENDS
  MLIRTensorRTRuntimeOpsIncGen

  LINK_LIBS PUBLIC
    MLIRExecutorCommonUtils
    MLIRTensorRTCUDADialect
    MLIRTensorRTStreamSchedulableOpInterface
    MLIRTensorRTTensorRTDialect
  MLIR_LIBS PUBLIC
    MLIRBufferizationDialect
    MLIRDialect
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRSupport
  )
