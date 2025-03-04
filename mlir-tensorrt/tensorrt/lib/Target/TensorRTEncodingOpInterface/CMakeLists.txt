add_mtrtd_library(MLIRTensorRTTensorRTEncodingOpInterface
  NetworkEncoder.cpp
  TensorRTEncodingOpInterface.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  TensorRTHeaderOnly
  CUDA::cudart
  MLIRTensorRTTensorRTUtils

  DEPENDS
  MLIRTensorRTEnumConverterGen
  MLIRTensorRTEncodingOpInterfaceIncGen
)
