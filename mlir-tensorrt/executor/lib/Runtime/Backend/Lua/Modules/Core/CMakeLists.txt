add_mlir_executor_runtime_capability_library(MLIRTensorRTExecutorRuntimeCoreModule
  CoreModule.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTExecutorRuntimeAPI

  LINK_LIBS PRIVATE
  sol2::sol2
  lua::core
  nvtx3-cpp
  MLIRTensorRTExecutorRuntimeCommon
  MLIRTensorRTExecutorRuntimeModuleUtils
  MLIRTensorRTSupportAllocators
  MLIRExecutorCAPIRuntimeCore
  MLIRTensorRTLuaExtensionRegistry
)

target_compile_options(obj.MLIRTensorRTExecutorRuntimeCoreModule
  PRIVATE
  "$<$<CXX_COMPILER_ID:GNU>:-Wno-strict-aliasing>"
)