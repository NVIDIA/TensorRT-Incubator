add_mlir_tensorrt_public_c_api_library(MLIRTensorRTCAPICommon
  Common.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_TENSORRT_ROOT_DIR}/include

  LINK_LIBS PUBLIC
  MLIRTensorRTExecutorRuntimeAPI
  MLIRTensorRTSupportStatus
)
