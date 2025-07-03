cmake_minimum_required(VERSION 3.25)
project(mlir-tensorrt-dialect LANGUAGES C CXX)

#-------------------------------------------------------------------------------
# Options and settings
#-------------------------------------------------------------------------------
set(MLIR_TRT_DIALECT_FEATURE_FLAGS)
macro(mtrt_dialect_option name)
  option(${name} ${ARGN})
  list(APPEND MLIR_TRT_DIALECT_FEATURE_FLAGS ${name})
endmacro()

mtrt_dialect_option(MLIR_TRT_DIALECT_TARGET_TENSORRT
  "Enable exporting TensorRT dialect IR to a TensorRT engine" ON)

#-------------------------------------------------------------------------------
# Project Setup
#-------------------------------------------------------------------------------

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 "C++ standard to conform to")

if(PROJECT_IS_TOP_LEVEL)
  message(STATUS "Building MLIR-Tensorrt-dialect as standalone project")
endif()

if(MLIR_ENABLE_BINDINGS_PYTHON AND NOT Python3_EXECUTABLE)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

include(cmake/TensorRTFunctions.cmake)

#-------------------------------------------------------------------------------
# Dependencies
#-------------------------------------------------------------------------------
find_package(MLIRTensorRTCommon REQUIRED)

if(NOT TARGET MLIRSupport)
  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include_directories(${MLIR_INCLUDE_DIRS})
endif()

if(PROJECT_IS_TOP_LEVEL)
  include(HandleLLVMOptions)
endif()

if(MLIR_TRT_DIALECT_TARGET_TENSORRT)
  find_package(CUDAToolkit REQUIRED)
endif()

set(MLIR_TENSORRT_DIALECT_LIB_DIR "${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}")
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR "${MLIR_TENSORRT_DIALECT_LIB_DIR}")

#-------------------------------------------------------------------------------
# Project Targets
#-------------------------------------------------------------------------------
set(MLIR_TRT_DIALECT_DEFINITIONS)
foreach(feature_flag IN LISTS MLIR_TRT_DIALECT_FEATURE_FLAGS)
  if(${feature_flag})
    list(APPEND MLIR_TRT_DIALECT_DEFINITIONS "-D${feature_flag}")
  endif()
endforeach()
add_definitions(${MLIR_TRT_DIALECT_DEFINITIONS})

set(MLIR_TENSORRT_DIALECT_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})
set(MLIR_TENSORRT_DIALECT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${MLIR_TENSORRT_DIALECT_SOURCE_DIR}/include)
include_directories(${MLIR_TENSORRT_DIALECT_BINARY_DIR}/include)

add_subdirectory(include/mlir-tensorrt-dialect)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  add_subdirectory(python)
endif()
