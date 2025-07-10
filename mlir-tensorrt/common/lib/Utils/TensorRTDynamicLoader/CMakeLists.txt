add_mlir_tensorrt_library(MLIRTRTTensorRTDynamicLoader
  TensorRTDynamicLoader.cpp

  LINK_LIBS PRIVATE
  CUDA::cudart
  TensorRTHeaderOnly
  MLIRSupport
)
