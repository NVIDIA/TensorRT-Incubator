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

target_include_directories(MLIRTensorRTExecutorRuntimeTensorRTModule
  PRIVATE
  ${MLIR_TENSORRT_ROOT_DIR}/tensorrt/include
)
