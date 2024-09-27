add_mlir_library(MLIRTensorRTTensorRTPluginRegistry
  PluginRegistry.cpp

  LINK_LIBS PUBLIC
  CUDA::cudart
  MLIRTRTTensorRTDynamicLoader
  TensorRTHeaderOnly
)
