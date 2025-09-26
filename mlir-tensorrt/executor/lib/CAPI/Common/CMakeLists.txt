mtrt_add_capi_library(MLIRTensorRTCAPICommon
  Common.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_TENSORRT_ROOT_DIR}/include

  LINK_LIBS PUBLIC
  MLIRTensorRTExecutorRuntimeAPI
  MLIRTensorRTCAPISupportStatus
  )
