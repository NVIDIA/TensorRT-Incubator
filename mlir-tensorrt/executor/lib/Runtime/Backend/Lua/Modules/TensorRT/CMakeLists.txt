add_mlir_library(MLIRTensorRTExecutorRuntimeTensorRTModule
  TensorRTModule.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTExecutorRuntimeAPI

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
