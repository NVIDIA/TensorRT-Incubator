add_mlir_public_c_api_library(MLIRExecutorCAPIRuntimeCore
  CoreModule.cpp

  PARTIAL_SOURCES_INTENDED
  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_LIST_DIR}

  LINK_LIBS PUBLIC
  MLIRTensorRTCAPISupportStatus
  )

add_mlir_public_c_api_library(MLIRExecutorCAPIRuntimeCUDA
  CUDAModule.cpp
  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_LIST_DIR}

  LINK_LIBS PRIVATE
  MLIRTensorRTCAPISupportStatus
  CUDA::cudart
  MLIRTensorRTExecutorPtxCompilerUtils
  MLIRTensorRTExecutorRuntimeCommon
  )

add_mlir_public_c_api_library(MLIRExecutorCAPIRuntimeTensorRT
  TensorRTModule.cpp
  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_LIST_DIR}

  LINK_LIBS PRIVATE
  MLIRTensorRTCAPISupportStatus
  CUDA::cudart
  MLIRTensorRTExecutorRuntimeCommon
  MLIRTRTTensorRTDynamicLoader
  TensorRTHeaderOnly
  )

# This is an aggregate library for mlir-runner style
# tests.
add_mlir_aggregate(mtrt_runtime
  MTRTRuntime.cpp

  SHARED
  EMBED_LIBS
    MLIRExecutorCAPIRuntimeCore
    MLIRExecutorCAPIRuntimeCUDA
    MLIRExecutorCAPIRuntimeTensorRT
  )