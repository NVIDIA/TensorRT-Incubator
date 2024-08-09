add_mlir_executor_library(MLIRTRTTensorRTDynamicLoader
  TensorRTDynamicLoader.cpp

  LINK_LIBS PRIVATE
  CUDA::cudart
  TensorRTHeaderOnly
  MLIRSupport
)
