cmake_minimum_required(VERSION 3.25)
project(mlir-tensorrt-common LANGUAGES CXX)

# Depdendencies
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
include(HandleLLVMOptions)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

if(MLIR_TRT_TARGET_TENSORRT)
  find_package(TensorRT REQUIRED)
endif()

find_package(CUDAToolkit REQUIRED)

set(MLIR_TENSORRT_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_TENSORRT_COMMON_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include_directories(include ${CMAKE_CURRENT_BINARY_DIR}/include)

add_library(MLIRTensorRTCommonIncludes INTERFACE)
target_include_directories(MLIRTensorRTCommonIncludes INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
)

add_subdirectory(include/mlir-tensorrt-common)
add_subdirectory(lib)

install(TARGETS MLIRTensorRTCommonIncludes
  EXPORT MLIRTensorRTCommonTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

