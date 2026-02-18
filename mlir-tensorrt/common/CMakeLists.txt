set(MLIR_TENSORRT_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_TENSORRT_COMMON_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

add_library(MLIRTensorRTCommonIncludes INTERFACE)
target_include_directories(MLIRTensorRTCommonIncludes INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
)
mtrt_add_install(MLIRTensorRTCommonIncludes UMBRELLA mtrt-headers)

add_subdirectory(tools/mtrt-tblgen)
add_subdirectory(mlir-tensorrt-common)

add_dependencies(MLIRTensorRTCommonIncludes MLIRTensorRTStatusEnumIncGen)

mtrt_add_header_installation_components(mtrt-headers
  DIRECTORIES
   "mlir-tensorrt-common"
   "${CMAKE_CURRENT_BINARY_DIR}/mlir-tensorrt-common"
)
