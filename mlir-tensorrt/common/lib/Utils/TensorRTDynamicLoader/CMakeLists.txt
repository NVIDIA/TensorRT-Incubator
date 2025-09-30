add_mlir_tensorrt_library(MLIRTRTTensorRTDynamicLoader
  TensorRTDynamicLoader.cpp

  LINK_LIBS PRIVATE
    ${MLIR_TRT_CUDA_TARGET}
    TensorRTHeaderOnly
  MLIR_LIBS PRIVATE
    MLIRSupport
)
