add_mlir_executor_library(MLIRTensorRTSupportDeviceInfo
  PARTIAL_SOURCES_INTENDED
  DeviceInfo.cpp

  LINK_LIBS PUBLIC
  ${MLIR_TRT_CUDA_TARGET}
  MLIRExecutorIncludes
  MLIRTensorRTSupportStatus
)

add_mlir_executor_library(MLIRTensorRTSupportStridedCopy
  PARTIAL_SOURCES_INTENDED
  StridedCopy.cpp

  LINK_LIBS PUBLIC
  MLIRExecutorIncludes
  MLIRTensorRTSupportStatus
)

add_mlir_executor_library(MLIRTensorRTSupportArtifactManager
  PARTIAL_SOURCES_INTENDED
  ArtifactManager.cpp

  LINK_LIBS PUBLIC
    MLIRExecutorSerializationUtils

  MLIR_LIBS PUBLIC
    MLIRIR
  )
