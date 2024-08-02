add_mlir_library(MLIRTensorRTExecutorRuntimeCommon
  CommonRuntime.cpp
  NvPtxCompilerUtils.cpp

  LINK_LIBS

  PUBLIC  
  MLIRTensorRTExecutorRuntimeAPI
  CUDA::cuda_driver

  PRIVATE
  CUDANVPTXCompilerLibraryPatched
)
