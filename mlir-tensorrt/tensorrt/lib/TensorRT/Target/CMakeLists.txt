add_mlir_library(MLIRTensorRTTensorRTEncodingImpl
  TensorRTEncodingImpl.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTensorRTDialect
  MLIRTensorRTTensorRTEncodingOpInterface
  MLIRTransformUtils

  DEPENDS
  MLIRTensorRTTensorRTEncodingIncGen
)
