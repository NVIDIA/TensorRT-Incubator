mtrt_add_capi_library(MLIRTensorRTCAPIRuntime
  Runtime.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTCAPICommon
  MLIRTensorRTCAPISupportStatus
  MLIRTensorRTExecutionEngineLuaRuntime
  MLIRTensorRTExecutorRuntimeAPI
  MLIRTensorRTExecutorRuntimeCommon
)
