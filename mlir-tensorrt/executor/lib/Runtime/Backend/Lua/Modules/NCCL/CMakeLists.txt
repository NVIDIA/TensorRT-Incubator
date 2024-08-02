
add_mlir_library(MLIRTensorRTExecutorRuntimeNCCLModule
  NCCLModule.cpp

  LINK_LIBS

  PUBLIC
  MLIRTensorRTExecutorRuntimeAPI

  PRIVATE
  sol2::sol2
  lua::core
  CUDA::cudart
  CUDA::cuda_driver
  $<$<BOOL:${MLIR_TRT_ENABLE_NCCL}>:MPI::MPI_C>
  $<$<BOOL:${MLIR_TRT_ENABLE_NCCL}>:NCCL>
  MLIRTensorRTExecutorRuntimeCommon
)
