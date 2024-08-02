add_mlir_library(MLIRTensorRTTensorRTPluginRegistry
  PluginRegistry.cpp

  LINK_LIBS PUBLIC
  TensorRTHeaderOnly
  CUDA::cudart
)
