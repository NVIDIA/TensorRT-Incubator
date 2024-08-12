add_mlir_executor_runtime_library(MLIRTensorRTRuntimeSupport
  MPI.cpp

  LINK_LIBS

  PUBLIC
  MLIRTensorRTSupportStatus

  PRIVATE
  $<$<BOOL:${MLIR_EXECUTOR_ENABLE_MPI}>:MPI::MPI_C>
)
