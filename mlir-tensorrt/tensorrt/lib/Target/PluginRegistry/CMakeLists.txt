add_mtrtd_library(MLIRTensorRTTensorRTPluginRegistry
  PluginRegistry.cpp

  LINK_LIBS PUBLIC
    ${MLIR_TRT_CUDA_TARGET}
    TensorRTHeaderOnly
  MLIR_LIBS PUBLIC
    TensorRTDynamicLoader
)
