set(MLIR_TENSORRT_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_TENSORRT_COMMON_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include_directories(include ${CMAKE_CURRENT_BINARY_DIR}/include)

add_library(MLIRTensorRTCommonIncludes INTERFACE)
target_include_directories(MLIRTensorRTCommonIncludes INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
)
mtrt_add_install(MLIRTensorRTCommonIncludes UMBRELLA mtrt-headers)

add_subdirectory(tools/mtrt-tblgen)
add_subdirectory(include/mlir-tensorrt-common)
add_subdirectory(lib)

add_dependencies(MLIRTensorRTCommonIncludes MLIRTensorRTStatusEnumIncGen)

mtrt_add_header_installation_components(mtrt-headers
  DIRECTORIES
   "include/mlir-tensorrt-common"
   "include/mlir-tensorrt-common-c"
   "${CMAKE_CURRENT_BINARY_DIR}/include/mlir-tensorrt-common"
   "${CMAKE_CURRENT_BINARY_DIR}/include/mlir-tensorrt-common-c"
)