cmake_minimum_required(VERSION 3.25)
project(mlir-executor LANGUAGES C CXX)

#-------------------------------------------------------------------------------
# Options and settings
#-------------------------------------------------------------------------------

macro(mlir_executor_option name)
  option(${name} ${ARGN})
  list(APPEND MLIR_EXECUTOR_FEATURE_FLAGS ${name})
endmacro()

mlir_executor_option(MLIR_EXECUTOR_ENABLE_CUBLAS "Enable use of CUBLAS in the runtime" ON)
mlir_executor_option(MLIR_EXECUTOR_ENABLE_NCCL "Enable use of NCCL in the runtime" ON)
mlir_executor_option(MLIR_EXECUTOR_ENABLE_MPI "Enable use of MPI in the runtime" ON)
mlir_executor_option(MLIR_EXECUTOR_ENABLE_TENSORRT "Enable TensorRT runtime module" ON)
mlir_executor_option(MLIR_EXECUTOR_ENABLE_CUDA "Enable CUDA runtime module" ON)
mlir_executor_option(MLIR_EXECUTOR_TARGET_LUA "Enable Lua translations and runtime backend" ON)
mlir_executor_option(MLIR_EXECUTOR_ENABLE_GPU_INTEGRATION_TESTS "Enable integration tests that require GPU" ON)

#-------------------------------------------------------------------------------
# Project Setup
#-------------------------------------------------------------------------------
include(cmake/ExecutorDependencies.cmake)
include(cmake/ExecutorFunctions.cmake)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(PROJECT_IS_TOP_LEVEL)
  message(STATUS "Building MLIR-Executor as standalone project")

  if(MLIR_DIR)
    find_package(MLIR REQUIRED CONFIG)
    message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
    set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
    set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
    set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  else()
    set(MLIR_EXECUTOR_LLVM_COMMIT "c8b5d30f707757a4fe4d9d0bb01f762665f6942f")
    mlir_executor_add_llvm_project(
      NAME llvm_project
      VERSION 0.0.20240827
      URL "https://github.com/llvm/llvm-project/archive/${MLIR_EXECUTOR_LLVM_COMMIT}.zip"
      EXCLUDE_FROM_ALL TRUE
      SOURCE_SUBDIR "llvm"
      OPTIONS
        "LLVM_ENABLE_PROJECTS mlir"
        "LLVM_TARGETS_TO_BUILD host"
        # Any other LLVM CMake flags should be set directly from the
        # CMake configuration command.
    )
  endif()
elseif(NOT PROJECT_IS_TOP_LEVEL AND NOT MLIR_BINARY_DIR)
  # Build via external projects mechanism
  message(STATUS "Building MLIR-Executor via LLVM_EXTERNAL_PROJECTS")
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
else()
  message(STATUS "Building MLIR-Executor as sub-project")
endif()

if(PROJECT_IS_TOP_LEVEL OR NOT COMMAND add_mlir_library)
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
endif()

if(MLIR_ENABLE_BINDINGS_PYTHON AND NOT Python3_EXECUTABLE)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

#-------------------------------------------------------------------------------
# Dependencies
#-------------------------------------------------------------------------------

find_package(CUDAToolkit REQUIRED)
if(MLIR_EXECUTOR_ENABLE_CUDA)
  mlir_executor_find_and_patch_libnvptxcompiler(CUDANVPTXCompilerLibraryPatched)
endif()

mlir_executor_add_nvtx()
mlir_executor_add_lua()
mlir_executor_add_sol2()
mlir_executor_add_flatbuffers()
mlir_tensorrt_find_dlpack()

if(MLIR_EXECUTOR_ENABLE_MPI)
  find_package(MPI COMPONENTS C)
endif()

if(MLIR_EXECUTOR_ENABLE_NCCL)
  mlir_executor_find_nccl()
endif()

if(MLIR_EXECUTOR_ENABLE_NCCL AND NOT TARGET NCCL)
  message(FATAL_ERROR "MLIR_EXECUTOR_ENABLE_NCCL=ON but NCCL was not found")
endif()
if(MLIR_EXECUTOR_ENABLE_MPI AND NOT MPI_FOUND)
  message(FATAL_ERROR "MLIR_EXECUTOR_ENABLE_MPI=ON but MPI was not found")
endif()

#-------------------------------------------------------------------------------
# Project Targets
#-------------------------------------------------------------------------------
set(MLIR_EXECUTOR_DEFINITIONS)
foreach(feature_flag IN LISTS MLIR_EXECUTOR_FEATURE_FLAGS)
  if(${feature_flag})
    list(APPEND MLIR_EXECUTOR_DEFINITIONS "-D${feature_flag}")
  endif()
endforeach()

set(MLIR_EXECUTOR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_EXECUTOR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${MLIR_EXECUTOR_SOURCE_DIR}/include)
include_directories(${MLIR_EXECUTOR_BINARY_DIR}/include)

if(PROJECT_IS_TOP_LEVEL)
  separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
  list(REMOVE_DUPLICATES LLVM_DEFINITIONS_LIST)
  link_directories(${LLVM_BUILD_LIBRARY_DIR})
  add_definitions(${LLVM_DEFINITIONS_LIST})
endif()

add_definitions(${MLIR_EXECUTOR_DEFINITIONS})

add_subdirectory(include/mlir-executor)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)

get_property(MLIR_EXECUTOR_LIBS GLOBAL PROPERTY MLIR_EXECUTOR_LIBS)
add_library(MLIRTensorRTExecutorLibraries INTERFACE)
target_link_libraries(MLIRTensorRTExecutorLibraries INTERFACE
  ${MLIR_EXECUTOR_LIBS})
add_library(MLIRTensorRT::Executor ALIAS MLIRTensorRTExecutorLibraries)
add_mlir_library_install(MLIRTensorRTExecutorLibraries)
