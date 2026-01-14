add_mlir_executor_runtime_capability_library(MLIRTensorRTExecutorRuntimeNCCLModule
  NCCLModule.cpp

  LINK_LIBS PUBLIC
  ${MLIR_TRT_CUDA_TARGET}
  ${MLIR_TRT_MPI_TARGET}
  ${MLIR_TRT_NCCL_TARGET}
  ${MLIR_TRT_CUDA_DRIVER_TARGET}
  lua::core
  MLIRTensorRTExecutorCUDAUtils
  MLIRTensorRTExecutorRuntimeAPI
  MLIRTensorRTExecutorRuntimeCommon
  MLIRTensorRTLuaExtensionRegistry
  )
