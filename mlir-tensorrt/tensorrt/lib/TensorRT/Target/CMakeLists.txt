add_mtrtd_library(MLIRTensorRTTensorRTEncodingImpl
  TensorRTEncodingImpl.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTensorRTTensorRTDialect
  MLIRTensorRTTensorRTEncodingOpInterface
  MLIRTransformUtils

  DEPENDS
  MLIRTensorRTTensorRTEncodingIncGen
)
