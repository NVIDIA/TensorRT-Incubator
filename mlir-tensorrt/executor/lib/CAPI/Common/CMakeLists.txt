mtrt_add_capi_library(MLIRTensorRTCAPICommon
  Common.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTExecutorRuntimeAPI
  MLIRTensorRTCAPISupportStatus
  )
