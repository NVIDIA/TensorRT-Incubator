add_mlir_tensorrt_library(MLIRTensorRTCUDAUtils
  CUDAUtils.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRTensorRTCUDADialect

  MLIR_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  )
