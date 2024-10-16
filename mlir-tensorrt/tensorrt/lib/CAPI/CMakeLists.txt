add_mlir_public_c_api_library(MLIRTensorRTCAPITensorRTDialect
  TensorRTAttributes.cpp
  TensorRTDialect.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTDialect
  MLIRTensorRTTransforms
)
