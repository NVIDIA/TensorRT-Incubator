find_package(CUDAToolkit REQUIRED)

add_mlir_library(MLIRTensorRTSupportStatus
  PARTIAL_SOURCES_INTENDED
  Status.cpp

  DEPENDS
  MLIRTensorRTStatusEnumIncGen
  MLIRTensorRTStatusEnumCIncGen
)

add_mlir_library(MLIRTensorRTSupportAllocators
  PARTIAL_SOURCES_INTENDED
  Allocators.cpp

  LINK_LIBS PUBLIC
  CUDA::cudart

  MLIRTensorRTSupportStatus
)

add_mlir_library(MLIRTensorRTSupportDeviceInfo
  PARTIAL_SOURCES_INTENDED
  DeviceInfo.cpp

  LINK_LIBS PUBLIC
  CUDA::cudart
  MLIRTensorRTSupportStatus
)
