add_mlir_executor_runtime_library(MLIRTensorRTRuntimeSupport
  MPI.cpp
  CUDAHelpers.cpp

  LINK_LIBS

  PUBLIC
  MLIRTensorRTSupportStatus

  PRIVATE
  $<$<BOOL:${MLIR_TRT_ENABLE_MPI}>:MPI::MPI_C>
  ${MLIR_TRT_CUDA_TARGET}
)
