mtrt_add_capi_library(MLIRTensorRTCAPIRuntime
  Runtime.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_TENSORRT_ROOT_DIR}/include

  LINK_LIBS PUBLIC
  MLIRTensorRTCAPICommon
  MLIRTensorRTCAPISupportStatus
  MLIRTensorRTExecutionEngineLuaRuntime
  MLIRTensorRTExecutorRuntimeAPI
  MLIRTensorRTExecutorRuntimeCommon
)
