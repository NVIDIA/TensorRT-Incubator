if(MLIR_EXECUTOR_ENABLE_TENSORRT)
  add_subdirectory(TensorRTDynamicLoader EXCLUDE_FROM_ALL)
endif()

add_mlir_executor_library(MLIRExecutorCommonUtils
  PARTIAL_SOURCES_INTENDED
  RegionUtils.cpp

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRTransformUtils
  MLIRPass
  MLIRSupport
  )

add_mlir_executor_library(MLIRExecutorSerializationUtils
  PARTIAL_SOURCES_INTENDED
  SerializationUtils.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDataLayoutInterfaces
  )
