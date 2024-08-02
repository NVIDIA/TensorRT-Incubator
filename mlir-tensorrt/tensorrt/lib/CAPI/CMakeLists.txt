add_mlir_public_c_api_library(MLIRTensorRTCAPITensorRTDialect
  TensorRTAttributes.cpp
  TensorRTDialect.cpp  

  ADDITIONAL_HEADER_DIRS
  ${MLIR_TENSORRT_ROOT_DIR}/include

  LINK_LIBS PUBLIC
  MLIRTensorRTDialect
  MLIRTensorRTTransforms
)
