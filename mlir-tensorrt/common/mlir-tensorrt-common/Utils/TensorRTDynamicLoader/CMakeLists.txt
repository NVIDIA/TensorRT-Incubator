add_mlir_tensorrt_library(TensorRTDynamicLoader
  TensorRTDynamicLoader.cpp

  LINK_LIBS PUBLIC
  TensorRTHeaderOnly
  ${MLIR_TRT_CUDA_TARGET}
  )
