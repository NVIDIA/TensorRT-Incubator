find_package(CUDAToolkit REQUIRED)

add_mlir_executor_library(MLIRTensorRTSupportStatus
  PARTIAL_SOURCES_INTENDED
  Status.cpp

  DEPENDS
  MLIRTensorRTStatusEnumIncGen
  MLIRTensorRTStatusEnumCIncGen
)

set(deps_ MLIRTensorRTSupportStatus)
if(MLIR_EXECUTOR_ENABLE_CUDA)
  list(APPEND deps_ CUDA::cudart)
endif()


add_mlir_executor_library(MLIRTensorRTSupportAllocators
  PARTIAL_SOURCES_INTENDED
  Allocators.cpp

  LINK_LIBS PUBLIC ${deps_}
)

add_mlir_executor_library(MLIRTensorRTSupportDeviceInfo
  PARTIAL_SOURCES_INTENDED
  DeviceInfo.cpp

  LINK_LIBS PUBLIC ${deps_}
)
