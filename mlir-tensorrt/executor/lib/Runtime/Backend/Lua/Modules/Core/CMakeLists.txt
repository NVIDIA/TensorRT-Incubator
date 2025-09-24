add_mlir_executor_runtime_capability_library(MLIRTensorRTExecutorRuntimeCoreModule
  CoreModule.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTExecutorRuntimeAPI

  LINK_LIBS PRIVATE
  lua::core
  MLIRExecutorCAPIRuntimeCore
  MLIRTensorRTExecutorRuntimeCommon
  MLIRTensorRTExecutorRuntimeModuleUtils
  MLIRTensorRTLuaExtensionRegistry
  MLIRTensorRTSupportAllocators
  MLIRTensorRTSupportStridedCopy
  nvtx3-cpp
  sol2::sol2
)

target_compile_options(obj.MLIRTensorRTExecutorRuntimeCoreModule
  PRIVATE
  "$<$<CXX_COMPILER_ID:GNU>:-Wno-strict-aliasing>"
)
