set(optional_deps)
if(MLIR_TRT_TARGET_TENSORRT)
  list(APPEND optional_deps TensorRTHeaderOnly TensorRTDynamicLoader)
endif()


mtrt_add_capi_library(MLIRTensorRTCAPITensorRTDialect
  TensorRTAttributes.cpp
  TensorRTDialect.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTTensorRTDialect
  MLIRTensorRTTransforms
  MLIRTensorRTDialectIncludes
  ${optional_deps}
)
