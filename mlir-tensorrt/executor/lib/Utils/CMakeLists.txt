if(MLIR_EXECUTOR_ENABLE_TENSORRT)
  add_subdirectory(TensorRTDynamicLoader EXCLUDE_FROM_ALL)
endif()

add_mlir_executor_library(MLIRExecutorCommonUtils
  PARTIAL_SOURCES_INTENDED
  RegionUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_TENSORRT_ROOT_DIR}/include

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRTransformUtils
  MLIRPass
  MLIRSupport
  )
