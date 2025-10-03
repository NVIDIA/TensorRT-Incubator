mtrt_add_project_library(MLIRExecutorCRuntimeCore
  PROJECT_NAME MLIRExecutor
  CoreModule.cpp

  PARTIAL_SOURCES_INTENDED
  ENABLE_AGGREGATION
  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_LIST_DIR}

  LINK_LIBS PUBLIC
    MLIRTensorRTCAPISupportStatus
    MLIRTensorRTSupportStatus
  )

set(built_modules MLIRExecutorCRuntimeCore)

if(MLIR_TRT_ENABLE_CUDA)
  mtrt_add_project_library(MLIRExecutorCRuntimeCUDA
    PROJECT_NAME MLIRExecutor
    CUDAModule.cpp
    PARTIAL_SOURCES_INTENDED
    ENABLE_AGGREGATION

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_LIST_DIR}

    LINK_LIBS PRIVATE
      ${MLIR_TRT_CUDA_TARGET}
      MLIRTensorRTCAPISupportStatus
      MLIRTensorRTExecutorPtxCompilerUtils
      MLIRTensorRTExecutorRuntimeCommon
    )
  list(APPEND built_modules MLIRExecutorCRuntimeCUDA)
endif()

if(MLIR_TRT_TARGET_TENSORRT)
  mtrt_add_project_library(MLIRExecutorCRuntimeTensorRT
    PROJECT_NAME MLIRExecutor
    TensorRTModule.cpp
    PARTIAL_SOURCES_INTENDED
    ENABLE_AGGREGATION

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_LIST_DIR}

    LINK_LIBS PRIVATE
      ${MLIR_TRT_CUDA_TARGET}
      MLIRTensorRTCAPISupportStatus
      MLIRTensorRTExecutorRuntimeCommon
      TensorRTDynamicLoader
      TensorRTHeaderOnly
    )
  list(APPEND built_modules MLIRExecutorCRuntimeTensorRT)
endif()

# This is an aggregate library for mlir-runner style
# tests.
add_mlir_aggregate(mtrt_runtime
  MTRTRuntime.cpp

  SHARED
  EMBED_LIBS
    ${built_modules}
    MLIRTensorRTCAPISupportStatus
  )
