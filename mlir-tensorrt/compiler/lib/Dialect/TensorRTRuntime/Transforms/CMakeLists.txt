add_mlir_tensorrt_library(MLIRTensorRTTensorRTRuntimeTransforms
  BufferizableOpInterfaceImpl.cpp

  LINK_LIBS
  MLIRIR
  MLIRTensorRTPlanDialect
  MLIRTensorRTTensorRTRuntimeDialect
  MLIRBufferizationDialect
)
