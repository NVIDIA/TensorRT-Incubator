add_mlir_executor_runtime_library(MLIRTensorRTRuntimeSupport
  MPI.cpp
  CUDAHelpers.cpp
  CUDAEventPool.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRTensorRTSupportStatus
  ${MLIR_TRT_MPI_TARGET}
  ${MLIR_TRT_CUDA_TARGET}
  )

add_mlir_executor_library(MLIRTensorRTSupportAllocators
  PARTIAL_SOURCES_INTENDED
  Allocators.cpp

  LINK_LIBS PUBLIC
  MLIRExecutorIncludes
  MLIRTensorRTRuntimeSupport
  MLIRTensorRTSupportStatus
)
