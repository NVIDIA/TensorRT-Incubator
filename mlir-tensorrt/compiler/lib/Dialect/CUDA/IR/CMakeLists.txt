add_mlir_tensorrt_dialect_library(MLIRTensorRTCUDADialect
  CUDA.cpp

  ADDITIONAL_HEADER_DIRS
  # TODO: we should remove requirement on executor.
  # This may not even be needed
  ${MLIR_TENSORRT_ROOT_DIR}/executor/include

  DEPENDS
  MLIRCUDAOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTensorRTPlanDialect
  MLIRTensorRTExecutorDialect
  )
