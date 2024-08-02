add_mlir_tensorrt_library(MLIRTensorRTCUDATransforms
  BufferizableOpInterfaceImpl.cpp

  LINK_LIBS
  MLIRIR
  MLIRTensorRTCUDADialect
  MLIRBufferizationDialect
)