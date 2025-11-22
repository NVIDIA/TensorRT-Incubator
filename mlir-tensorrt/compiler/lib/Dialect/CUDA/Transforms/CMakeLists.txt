add_mlir_tensorrt_library(MLIRTensorRTCUDATransforms
  BufferizableOpInterfaceImpl.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTCUDADialect
  MLIR_LIBS PUBLIC
    MLIRIR
    MLIRBufferizationDialect
)