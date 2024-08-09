add_mlir_executor_runtime_capability_library(MLIRTensorRTExecutorRuntimeCuBLASModule
  CuBLASModule.cpp

  LINK_LIBS

  PUBLIC
  MLIRTensorRTExecutorRuntimeAPI

  PRIVATE
  MLIRTensorRTExecutorRuntimeCommon
  sol2::sol2
  lua::core
  nvtx3-cpp
  CUDA::cudart
  # TODO: link against static lib or shared stub library instead of linking cuBLAS shared lib directly
  # in order to be compatible with manylinux standard
  # for distiribution as pip package, see https://github.com/pypa/manylinux
  CUDA::cublasLt
  CUDA::cublas
)