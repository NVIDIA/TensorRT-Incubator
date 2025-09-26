mtrt_add_capi_library(MLIRExecutorCAPIRuntimeCore
  CoreModule.cpp

  PARTIAL_SOURCES_INTENDED
  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_LIST_DIR}

  LINK_LIBS PUBLIC
  MLIRTensorRTCAPISupportStatus
  )

set(built_modules MLIRExecutorCAPIRuntimeCore)

if(MLIR_TRT_ENABLE_CUDA)
  mtrt_add_capi_library(MLIRExecutorCAPIRuntimeCUDA
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
  list(APPEND built_modules MLIRExecutorCAPIRuntimeCUDA)
endif()

if(MLIR_TRT_TARGET_TENSORRT)
  mtrt_add_capi_library(MLIRExecutorCAPIRuntimeTensorRT
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
  list(APPEND built_modules MLIRExecutorCAPIRuntimeTensorRT)
endif()

# This is an aggregate library for mlir-runner style
# tests.
add_mlir_aggregate(mtrt_runtime
  MTRTRuntime.cpp

  SHARED
  EMBED_LIBS
    ${built_modules}
  )
