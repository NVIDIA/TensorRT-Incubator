cmake_minimum_required(VERSION 3.25)
project(mlir-tensorrt LANGUAGES CXX C)

# -------------------------------------------------
# Set CMake behavior policies
# -------------------------------------------------
include(build_tools/cmake/CMakePolicy.cmake NO_POLICY_SCOPE)

# -------------------------------------------------
# Package cache configuration
# -------------------------------------------------
if(NOT CPM_SOURCE_CACHE)
  message(WARNING "CPM_SOURCE_CACHE is not set. Source code for third party C++ packages will be \
  stored in your build directory. It is highly recommended to specify a CPM source cache directory outside of \
  your build directory (for example '$PWD/.cache.cpm')")
endif()

# -------------------------------------------------
# Include utils/tools
# -------------------------------------------------
include(build_tools/cmake/CPM.cmake)
include(build_tools/cmake/CompilationOptions.cmake)
include(build_tools/cmake/Targets.cmake)
include(build_tools/cmake/Dependencies.cmake)
include(Version.cmake)

# -------------------------------------------------
# Project Options / Configuration
# -------------------------------------------------
set(MLIR_TRT_FEATURE_FLAGS)
macro(mtrt_option name)
  option(${name} ${ARGN})
  list(APPEND MLIR_TRT_FEATURE_FLAGS ${name})
endmacro()

mtrt_option(MLIR_TRT_ENABLE_HLO "Whether to include stablehlo features" ON)
mtrt_option(MLIR_TRT_ENABLE_ASSERTIONS "Enables assertions" ON)
mtrt_option(MLIR_TRT_TARGET_TENSORRT "Enable exporting TensorRT dialect IR to a TensorRT engine" ON)
mtrt_option(MLIR_TRT_TARGET_CPP "Enable exporting TensorRT dialect IR to C++ code that calls the TensorRT API" ON)
mtrt_option(MLIR_TRT_ENABLE_PYTHON "Enable building the mlir_tensorrt_compiler python package." ON)
mtrt_option(MLIR_TRT_ENABLE_NVTX "Enable NVTX tracing" ON)
mtrt_option(MLIR_TRT_ENABLE_TESTING "Enable building optional tests" ON)
mtrt_option(MLIR_TRT_TARGET_LUA "Enable translating MLIR to the Lua target" ON)
mtrt_option(MLIR_TRT_ENABLE_EXECUTOR "Build the Executor dialect and MLIR-TensorRT Execution Engine" ON)
mtrt_option(MLIR_TRT_ENABLE_NCCL "Enable the NCCL runtime module" ON)
mtrt_option(MLIR_TRT_ENABLE_TORCH "Whether to include torch-mlir features" OFF)

set(MLIR_TRT_TENSORRT_DIR "" CACHE STRING "Path to TensorRT install directory")
set(MLIR_TRT_DOWNLOAD_TENSORRT_VERSION "10.9" CACHE STRING
   "Version of TensorRT to download and use. It overrides MLIR_TRT_TENSORRT_DIR.")
set(MLIR_TRT_PACKAGE_CACHE_DIR "" CACHE STRING "Directory where to cache downloaded C++ packages")
set(MLIR_TRT_USE_LINKER "" CACHE STRING "Specify a linker to use (e.g. LLD); this is just an alias for LLVM_USE_LINKER")

# Configure dependent Executor options.
set(MLIR_EXECUTOR_ENABLE_NCCL ${MLIR_TRT_ENABLE_NCCL} CACHE BOOL "" FORCE)
set(MLIR_EXECUTOR_ENABLE_MPI ${MLIR_TRT_ENABLE_NCCL} CACHE BOOL "" FORCE)
set(MLIR_EXECUTOR_ENABLE_CUBLAS ${MLIR_TRT_ENABLE_CUBLAS} CACHE BOOL "" FORCE)

# List of dialects to embed in the Python package. This is different between internal
# and public versions of the code.
set(MLIR_TRT_PYTHON_UPSTREAM_DIALECTS_EMBED
  affine arith bufferization builtin complex func math quant scf
  tensor
  )

# -------------------------------------------------
# Option validation
# -------------------------------------------------

# Some overly-complex logic for making `MLIR_TRT_USE_LINKER` alias `LLVM_USE_LINKER`.
if((MLIR_TRT_USE_LINKER AND LLVM_USE_LINKER) AND NOT
  (MLIR_TRT_USE_LINKER STREQUAL LLVM_USE_LINKER))
  message(FATAL_ERROR "MLIR_TRT_USE_LINKER=${MLIR_TRT_USE_LINKER} but "
                      "LLVM_USE_LINKER=${LLVM_USE_LINKER}. If both are set, they "
                      "must be equal.")
endif()
if(LLVM_USE_LINKER AND NOT MLIR_TRT_USE_LINKER)
  set(MLIR_TRT_USE_LINKER "${LLVM_USE_LINKER}" CACHE STRING "" FORCE)
endif()
if(MLIR_TRT_USE_LINKER AND NOT LLVM_USE_LINKER)
  set(LLVM_USE_LINKER "${MLIR_TRT_USE_LINKER}" CACHE STRING "" FORCE)
endif()

# Setup use of ccache for C, CXX, and CUDA languages if we are the top-level
# project.
# Enables generation of compile_commands.json for clangd.
if(PROJECT_IS_TOP_LEVEL)
  include(build_tools/cmake/CompilerCache.cmake)
  mtrt_use_compiler_cache()
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# -------------------------------------------------
# Option validation
# -------------------------------------------------
# Write out a header file containing convenience macros for each flag.
function(mtrt_write_feature_flags_header)
  set(feature_flags_header
    "${CMAKE_CURRENT_BINARY_DIR}/include/mlir-tensorrt/Features.h")

  # Generate the header at configure time
  file(WRITE "${feature_flags_header}" [[
  // Auto-generated feature macros, do not edit.
  #ifndef MLIR_TENSORRT_FEATURES_H
  #define MLIR_TENSORRT_FEATURES_H

  ]])

  foreach(FEATURE IN LISTS MLIR_TRT_FEATURE_FLAGS)
      file(APPEND "${feature_flags_header}" "#ifdef ${FEATURE}\n")
      file(APPEND "${feature_flags_header}" "#define IF_${FEATURE}(code) do { code } while (0)\n")
      file(APPEND "${feature_flags_header}" "#else\n")
      file(APPEND "${feature_flags_header}" "#define IF_${FEATURE}(code) do {} while (0)\n")
      file(APPEND "${feature_flags_header}" "#endif // ${FEATURE}\n\n")
  endforeach()
  file(APPEND "${feature_flags_header}" "#endif // MLIR_TENSORRT_FEATURES_H\n")
endfunction()

mtrt_write_feature_flags_header()
include_directories("${CMAKE_CURRENT_BINARY_DIR}/include")

# -------------------------------------------------
# Setup LLVM/MLIR
# -------------------------------------------------
find_package(Python3 REQUIRED
  COMPONENTS Interpreter Development)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

if(PROJECT_IS_TOP_LEVEL)
  set(MLIR_TRT_EXTERNAL_PROJECT_BUILD OFF)
  message("Building MLIR-TensorRT as a standalone project.")

  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
elseif(NOT MLIR_BINARY_DIR)
  set(MLIR_TRT_EXTERNAL_PROJECT_BUILD ON)
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
  message("Building MLIR-TensorRT as an LLVM external project.")
else()
  message("Building MLIR-TensorRT as a sub-project.")
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/lib)
endif()

# -------------------------------------------------
# Set global options
# -------------------------------------------------
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_C_STANDARD 11 CACHE STRING "C standard to conform to")

set(MLIR_TENSORRT_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})
set(MLIR_TENSORRT_ROOT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Sets up LLVM/MLIR build system and LLVM will override
# the global build/linker options.
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# -------------------------------------------------
# Dependencies
# -------------------------------------------------

# Download Stablehlo if it isn't provided by a parent project.
if(MLIR_TRT_ENABLE_HLO)
  find_package(Stablehlo REQUIRED)
  include_directories(${STABLEHLO_INCLUDE_DIRS})
endif()

if(MLIR_TRT_ENABLE_TORCH)
  find_package(torch_mlir REQUIRED)
endif()

if(MLIR_TRT_TARGET_TENSORRT)
  find_package(TensorRT REQUIRED)
endif()

if(MLIR_TRT_ENABLE_PYTHON)
  mlir_tensorrt_find_dlpack()
endif()

find_package(MLIRTensorRTCommon REQUIRED)

#--------------------------------------------------
# Diagnostics
#--------------------------------------------------

if(PROJECT_IS_TOP_LEVEL)
  message(STATUS "MLIR-TensorRT Project-Scope C/CXX/LINKER flags:")
  message(STATUS "-- CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
  message(STATUS "-- CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")
  message(STATUS "-- CMAKE_MODULE_LINKER_FLAGS: ${CMAKE_MODULE_LINKER_FLAGS}")
  message(STATUS "-- CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS}")
endif()

# -------------------------------------------------
# Target Directories
# -------------------------------------------------

set(MLIR_TRT_DEFINITIONS)
foreach(feature_flag IN LISTS MLIR_TRT_FEATURE_FLAGS)
  if(${feature_flag})
    list(APPEND MLIR_TRT_DEFINITIONS "-D${feature_flag}")
  endif()
endforeach()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS} ${MLIR_TRT_DEFINITIONS})

add_compile_options(-Wno-deprecated-declarations)

include(AddMLIRPython)
include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

# Declare the Python source targets ahead-of-time since the sources may be built
# up across multiple sub-directories.
if(MLIR_TRT_ENABLE_PYTHON)
  declare_mlir_python_sources(MLIRTensorRTPythonCompiler)
  declare_mlir_python_sources(MLIRTensorRTPythonCompiler.Dialects
    ADD_TO_PARENT MLIRTensorRTPythonCompiler)
endif()

# Add a meta target for all documentation generation targets. You can generate
# all documentation under `${buildDir}/docs` by building this target.
add_custom_target("mlir-tensorrt-doc")

# Add a meta target for running all sub-project `check-` targets.
add_custom_target(check-all-mlir-tensorrt)
add_custom_target(check-all-mlir-tensorrt-build-only)

if(MLIR_TRT_ENABLE_EXECUTOR)
  add_subdirectory(executor EXCLUDE_FROM_ALL)
endif()

add_subdirectory(tensorrt EXCLUDE_FROM_ALL)
include_directories(${CMAKE_CURRENT_LIST_DIR}/tensorrt/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/tensorrt/include)

add_subdirectory(compiler)
add_subdirectory(integrations)
