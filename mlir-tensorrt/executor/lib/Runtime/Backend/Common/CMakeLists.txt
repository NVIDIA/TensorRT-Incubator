add_mlir_executor_runtime_library(MLIRTensorRTExecutorRuntimeCommon
  CommonRuntime.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS

  PUBLIC
  MLIRTensorRTExecutorRuntimeAPI
)

if(MLIR_EXECUTOR_ENABLE_CUDA)

  add_mlir_executor_runtime_library(MLIRTensorRTExecutorPtxCompilerUtils
    NvPtxCompilerUtils.cpp
    PARTIAL_SOURCES_INTENDED

    LINK_LIBS

    PUBLIC
    MLIRTensorRTExecutorRuntimeAPI
    CUDA::cuda_driver

    PRIVATE
    CUDANVPTXCompilerLibraryPatched
  )

  add_mlir_executor_runtime_library(MLIRTensorRTExecutorCUDAUtils
    CUDACommon.cpp
    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
    MLIRTensorRTExecutorRuntimeAPI
    MLIRTensorRTExecutorPtxCompilerUtils
  )
endif()