add_mlir_tensorrt_library(MLIRTensorRTTensorRTRuntimeTransforms
  BufferizableOpInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTensorRTPlanDialect
  MLIRTensorRTTensorRTRuntimeDialect
  MLIRBufferizationDialect
)
