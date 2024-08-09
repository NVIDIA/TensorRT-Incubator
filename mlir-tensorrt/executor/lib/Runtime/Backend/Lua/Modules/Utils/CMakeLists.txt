add_mlir_executor_runtime_capability_library(MLIRTensorRTExecutorRuntimeModuleUtils
  MemRefUtils.cpp

  LINK_LIBS

  PUBLIC
  MLIRTensorRTExecutorRuntimeAPI

  PRIVATE
  sol2::sol2
  lua::core
  MLIRTensorRTExecutorRuntimeCommon
)
