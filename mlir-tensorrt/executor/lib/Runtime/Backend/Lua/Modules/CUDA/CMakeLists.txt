add_mlir_executor_runtime_capability_library(MLIRTensorRTExecutorRuntimeCUDAModule
  CUDAModule.cpp

  LINK_LIBS

  PUBLIC
  MLIRTensorRTExecutorRuntimeAPI
  MLIRTensorRTExecutorCUDAUtils

  PRIVATE
  MLIRTensorRTExecutorRuntimeCommon
  sol2::sol2
  lua::core
  nvtx3-cpp
  CUDA::cudart
  CUDA::cuda_driver
  MLIRTensorRTLuaExtensionRegistry
)

target_compile_options(MLIRTensorRTExecutorRuntimeCUDAModule
  PRIVATE
  "$<$<CXX_COMPILER_ID:GNU>:-Wno-strict-aliasing>"
)