add_mtrtd_library(MLIRTensorRTTensorRTEncodingImpl
  TensorRTEncodingImpl.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTTensorRTDialect
    MLIRTensorRTTensorRTEncodingOpInterface
  MLIR_LIBS PUBLIC
    MLIRIR
    MLIRTransformUtils
  DEPENDS
  MLIRTensorRTTensorRTEncodingIncGen
)
