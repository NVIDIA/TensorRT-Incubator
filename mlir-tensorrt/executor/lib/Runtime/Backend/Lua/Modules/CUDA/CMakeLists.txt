add_mlir_executor_runtime_capability_library(MLIRTensorRTExecutorRuntimeCUDAModule
  CUDAModule.cpp

  LINK_LIBS

  PUBLIC
  MLIRTensorRTExecutorRuntimeAPI
  MLIRTensorRTExecutorCUDAUtils

  PRIVATE
  CUDA::cuda_driver
  CUDA::cudart
  lua::core
  MLIRTensorRTExecutorRuntimeCommon
  MLIRTensorRTLuaExtensionRegistry
  MLIRTensorRTSupportStridedCopy
  nvtx3-cpp
  sol2::sol2
)

target_compile_options(MLIRTensorRTExecutorRuntimeCUDAModule
  PRIVATE
  "$<$<CXX_COMPILER_ID:GNU>:-Wno-strict-aliasing>"
)
