add_mlir_executor_runtime_capability_library(MLIRTensorRTExecutorRuntimeNCCLModule
  NCCLModule.cpp

  LINK_LIBS

  PUBLIC
  MLIRTensorRTExecutorRuntimeAPI
  MLIRTensorRTExecutorCUDAUtils

  PRIVATE
  sol2::sol2
  lua::core
  CUDA::cudart
  CUDA::cuda_driver
  $<$<BOOL:${MLIR_EXECUTOR_ENABLE_NCCL}>:MPI::MPI_C>
  $<$<BOOL:${MLIR_EXECUTOR_ENABLE_NCCL}>:NCCL>
  MLIRTensorRTExecutorRuntimeCommon
  MLIRTensorRTLuaExtensionRegistry
)
