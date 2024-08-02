add_mlir_library(MLIRTensorRTRuntimeSupport
  MPI.cpp

  LINK_LIBS

  PUBLIC
  MLIRTensorRTSupportStatus

  PRIVATE
  $<$<BOOL:${MLIR_TRT_ENABLE_NCCL}>:MPI::MPI_C>
)
