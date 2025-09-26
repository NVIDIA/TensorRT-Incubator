mtrt_add_capi_library(MLIRTensorRTCAPITensorRTDialect
  TensorRTAttributes.cpp
  TensorRTDialect.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTTensorRTDialect
  MLIRTensorRTTransforms
  MLIRTensorRTDialectIncludes
)
