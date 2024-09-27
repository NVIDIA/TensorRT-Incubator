add_mlir_public_c_api_library(MLIRTensorRTCAPISupportStatus
  Status.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_TENSORRT_ROOT_DIR}/include
  ${MLIR_TENSORRT_ROOT_BINARY_DIR}/include

  LINK_LIBS PUBLIC
  MLIRTensorRTSupportStatus

  DEPENDS
  MLIRTensorRTStatusEnumCIncGen
  )
