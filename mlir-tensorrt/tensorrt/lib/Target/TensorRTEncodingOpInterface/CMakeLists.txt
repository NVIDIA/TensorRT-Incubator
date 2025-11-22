add_mtrtd_library(MLIRTensorRTTensorRTEncodingOpInterface
  NetworkEncoder.cpp
  TensorRTEncodingOpInterface.cpp

  LINK_LIBS PUBLIC
    TensorRTHeaderOnly
    ${MLIR_TRT_CUDA_TARGET}
    MLIRTensorRTTensorRTUtils
  MLIR_LIBS PUBLIC
    MLIRIR
  DEPENDS
  MLIRTensorRTEnumConverterGen
  MLIRTensorRTEncodingOpInterfaceIncGen
)
