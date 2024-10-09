add_mlir_tensorrt_dialect_library(MLIRTensorRTTensorRTRuntimeDialect
  TensorRTRuntime.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_TENSORRT_ROOT_DIR}/include

  DEPENDS
  MLIRTensorRTRuntimeOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDialect
  MLIRSupport
  MLIRFuncDialect
  MLIRTensorRTDialect
  MLIRTensorRTCUDADialect
  MLIRExecutorCommonUtils
  )
