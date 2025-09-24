set(deps_ MLIRTensorRTSupportStatus)
if(MLIR_TRT_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
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

  LINK_LIBS PUBLIC
  ${deps_}
  MLIRExecutorIncludes
)

add_mlir_executor_library(MLIRTensorRTSupportStridedCopy
  PARTIAL_SOURCES_INTENDED
  StridedCopy.cpp

  LINK_LIBS PUBLIC
  MLIRExecutorIncludes
)
