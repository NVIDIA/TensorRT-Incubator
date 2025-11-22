add_mlir_executor_runtime_capability_library(MLIRTensorRTExecutorRuntimeTensorRTModule
  TensorRTModule.cpp

  LINK_LIBS PUBLIC
  lua::core
  MLIRTensorRTCommonIncludes
  MLIRTensorRTExecutorCUDAUtils
  MLIRTensorRTExecutorRuntimeAPI
  MLIRTensorRTExecutorRuntimeCommon
  MLIRTensorRTLuaExtensionRegistry
  MLIRTensorRTSupportAllocators
  TensorRTDynamicLoader
  nvtx3-cpp
  TensorRTHeaderOnly
)
