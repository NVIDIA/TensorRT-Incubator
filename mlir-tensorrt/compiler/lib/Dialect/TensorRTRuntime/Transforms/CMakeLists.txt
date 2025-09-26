add_mlir_tensorrt_library(MLIRTensorRTTensorRTRuntimeTransforms
  BufferizableOpInterfaceImpl.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTPlanDialect
    MLIRTensorRTTensorRTRuntimeDialect
  MLIR_LIBS PUBLIC
    MLIRIR
    MLIRBufferizationDialect
)
