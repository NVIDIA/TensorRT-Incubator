add_mlir_executor_runtime_library(MLIRTensorRTExecutorRuntimeCommon
  DataTypes.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRTensorRTSupportStatus
)

if(MLIR_TRT_ENABLE_CUDA)
  add_mlir_executor_runtime_library(MLIRTensorRTExecutorPtxCompilerUtils
    NvPtxCompilerUtils.cpp
    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
    CUDA::cuda_driver
    CUDANVPTXCompilerLibraryPatched
    MLIRTensorRTSupportStatus
  )

  add_mlir_executor_runtime_library(MLIRTensorRTExecutorCUDAUtils
    CUDACommon.cpp
    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
    MLIRTensorRTSupportStatus
    MLIRTensorRTExecutorPtxCompilerUtils
    MLIRTensorRTExecutorRuntimeAPI
  )
endif()
