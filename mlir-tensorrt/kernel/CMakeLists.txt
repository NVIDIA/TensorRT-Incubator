include(cmake/KernelFunctions.cmake)

set(MLIR_KERNEL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_KERNEL_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# Required for header ADT extra utilities.
include_directories(${CMAKE_CURRENT_LIST_DIR}/../executor/include)
get_target_property(common_include_dirs
                    MLIRTensorRTCommonIncludes INTERFACE_INCLUDE_DIRECTORIES)
include_directories(${common_include_dirs})

add_library(MLIRTensorRTKernelIncludes INTERFACE)
target_include_directories(MLIRTensorRTKernelIncludes INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
  )
mtrt_add_install(MLIRTensorRTKernelIncludes UMBRELLA mtrt-headers)

add_subdirectory(include/mlir-kernel)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)

mtrt_add_header_installation_components(mtrt-headers
  DIRECTORIES
   "include/mlir-kernel"
   "include/mlir-kernel-c"
   "${CMAKE_CURRENT_BINARY_DIR}/include/mlir-kernel"
   "${CMAKE_CURRENT_BINARY_DIR}/include/mlir-kernel-c"
  )
