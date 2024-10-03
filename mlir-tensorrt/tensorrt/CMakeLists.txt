cmake_minimum_required(VERSION 3.25)
project(mlir-tensorrt-dialect LANGUAGES C CXX)

#-------------------------------------------------------------------------------
# Options and settings
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Project Setup
#-------------------------------------------------------------------------------

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

if(PROJECT_IS_TOP_LEVEL)
  message(STATUS "Building MLIR-Tensorrt-dialect as standalone project")
  find_package(MLIR REQUIRED CONFIG)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
elseif(NOT PROJECT_IS_TOP_LEVEL AND NOT MLIR_BINARY_DIR)
  # Build via external projects mechanism
  message(STATUS "Building MLIR-Tensorrt-dialect via LLVM_EXTERNAL_PROJECTS")
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
else()
  message(STATUS "Building MLIR-tensorrt-dialect as sub-project")
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/lib)
endif()

if(PROJECT_IS_TOP_LEVEL OR NOT COMMAND mlir_add_library)
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
  separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
  list(REMOVE_DUPLICATES LLVM_DEFINITIONS_LIST)
  link_directories(${LLVM_BUILD_LIBRARY_DIR})
  add_definitions(${LLVM_DEFINITIONS_LIST})
endif()

if(MLIR_ENABLE_BINDINGS_PYTHON AND NOT Python3_EXECUTABLE)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

include(cmake/TensorRTFunctions.cmake)

#-------------------------------------------------------------------------------
# Dependencies
#-------------------------------------------------------------------------------

if(MLIR_TRT_TARGET_TENSORRT)
  find_package(CUDAToolkit REQUIRED)
endif()

#-------------------------------------------------------------------------------
# Project Targets
#-------------------------------------------------------------------------------
if(MLIR_TRT_TARGET_TENSORRT)
  add_definitions(-DMLIR_TRT_TARGET_TENSORRT)
endif()

set(MLIR_TENSORRT_DIALECT_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})
set(MLIR_TENSORRT_DIALECT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${MLIR_TENSORRT_DIALECT_SOURCE_DIR}/include)
include_directories(${MLIR_TENSORRT_DIALECT_BINARY_DIR}/include)

add_subdirectory(include/mlir-tensorrt-dialect)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tensorrt-opt)
