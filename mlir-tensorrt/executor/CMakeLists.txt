cmake_minimum_required(VERSION 3.25)
project(mlir-executor LANGUAGES C CXX)

#-------------------------------------------------------------------------------
# Options and settings
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Project Setup
#-------------------------------------------------------------------------------

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

if(PROJECT_IS_TOP_LEVEL)
  message(STATUS "Building MLIR-Executor as standalone project")
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
  message(STATUS "Building MLIR-Executor via LLVM_EXTERNAL_PROJECTS")
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
else()
  message(STATUS "Building MLIR-Executor as sub-project")
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/lib)
endif()

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

include(cmake/ExecutorDependencies.cmake)
include(cmake/ExecutorFunctions.cmake)

#-------------------------------------------------------------------------------
# Dependencies
#-------------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

mlir_executor_add_nvtx()
mlir_executor_add_lua()
mlir_executor_add_sol2()
mlir_executor_add_flatbuffers()
mlir_executor_find_and_patch_libnvptxcompiler(CUDANVPTXCompilerLibraryPatched)

if(MLIR_TRT_ENABLE_NCCL)
  find_package(MPI REQUIRED COMPONENTS C)
  mlir_executor_find_nccl()
endif()

#-------------------------------------------------------------------------------
# Project Targets
#-------------------------------------------------------------------------------
if(MLIR_TRT_ENABLE_NCCL)
  add_definitions(-DMLIR_TRT_ENABLE_NCCL)
endif()
if(MLIR_TRT_TARGET_LUA)
  add_definitions(-DMLIR_TRT_TARGET_LUA)
endif()
add_definitions(-DMLIR_TRT_ENABLE_NVTX)

set(MLIR_EXECUTOR_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(MLIR_EXECUTOR_BINARY_DIR ${PROJECT_BINARY_DIR})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${MLIR_EXECUTOR_SOURCE_DIR}/include)
include_directories(${MLIR_EXECUTOR_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(executor-tblgen)
add_subdirectory(include/mlir-executor)
add_subdirectory(lib)
add_subdirectory(executor-opt)
add_subdirectory(executor-translate)
add_subdirectory(executor-runner)
add_subdirectory(test)