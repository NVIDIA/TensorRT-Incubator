add_mlir_executor_runtime_capability_library(MLIRTensorRTExecutorRuntimeCUDAModule
  CUDAModule.cpp

  LINK_LIBS

  PUBLIC
  MLIRTensorRTExecutorRuntimeAPI
  MLIRTensorRTExecutorCUDAUtils

  PRIVATE
  ${MLIR_TRT_CUDA_DRIVER_TARGET}
  ${MLIR_TRT_CUDA_TARGET}
  lua::core
  MLIRTensorRTExecutorRuntimeCommon
  MLIRTensorRTLuaExtensionRegistry
  MLIRTensorRTSupportStridedCopy
  nvtx3-cpp
)

target_compile_options(MLIRTensorRTExecutorRuntimeCUDAModule
  PRIVATE
  "$<$<CXX_COMPILER_ID:GNU>:-Wno-strict-aliasing>"
)
