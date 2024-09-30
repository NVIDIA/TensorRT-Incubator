add_mlir_executor_runtime_capability_library(MLIRTensorRTExecutorRuntimeTensorRTModule
  TensorRTModule.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTExecutorRuntimeAPI
  MLIRTensorRTExecutorCUDAUtils

  PRIVATE
  MLIRTensorRTExecutorRuntimeCommon
  MLIRTensorRTExecutorRuntimeModuleUtils
  MLIRTRTTensorRTDynamicLoader
  MLIRTensorRTSupportAllocators
  sol2::sol2
  lua::core
  nvtx3-cpp
  TensorRTHeaderOnly
)
