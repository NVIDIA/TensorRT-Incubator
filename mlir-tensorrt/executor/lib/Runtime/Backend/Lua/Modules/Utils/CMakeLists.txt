add_mlir_library(MLIRTensorRTExecutorRuntimeModuleUtils
  MemRefUtils.cpp

  LINK_LIBS

  PUBLIC
  MLIRTensorRTExecutorRuntimeAPI

  PRIVATE
  sol2::sol2
  lua::core
  MLIRTensorRTExecutorRuntimeCommon
)
