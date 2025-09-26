add_mtrtd_library(MLIRTensorRTTensorRTEncodingOpInterface
  NetworkEncoder.cpp
  TensorRTEncodingOpInterface.cpp

  LINK_LIBS PUBLIC
    TensorRTHeaderOnly
    CUDA::cudart
    MLIRTensorRTTensorRTUtils
  MLIR_LIBS PUBLIC
    MLIRIR
  DEPENDS
  MLIRTensorRTEnumConverterGen
  MLIRTensorRTEncodingOpInterfaceIncGen
)
