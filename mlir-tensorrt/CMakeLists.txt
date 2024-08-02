cmake_minimum_required(VERSION 3.25)
project(mlir-tensorrt LANGUAGES CXX C)

include(Version.cmake)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

if(PROJECT_IS_TOP_LEVEL)
  set(MLIR_TRT_EXTERNAL_PROJECT_BUILD OFF)
  message("Building MLIR-TensorRT as a standalone project.")

  find_package(MLIR REQUIRED CONFIG)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

  # For whatever reason, StableHLO uses this CMake variable to decide whether to build
  # in "external project mode" or not. Since we don't want external project mode,
  # set it to the current binary dir.
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

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

# Setup use of ccache for C, CXX, and CUDA languages if we are the top-level
# project.
# Enables generation of compile_commands.json for clangd.
if(PROJECT_IS_TOP_LEVEL)
  include(build_tools/cmake/CompilerCache.cmake)
  mtrt_use_compiler_cache()
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# -------------------------------------------------
# Set CMake behavior policies
# -------------------------------------------------
if(POLICY CMP0135)
  # Ensure correct build behavior when downloaded archives
  # are involved.
  cmake_policy(SET CMP0135 NEW)
endif()

# -------------------------------------------------
# Project Options / Configuration
# -------------------------------------------------
option(MLIR_TRT_ENABLE_HLO "Whether to include stablehlo features" ON)
option(MLIR_TRT_ENABLE_ASSERTIONS "Enables assertions" ON)
option(MLIR_TRT_TARGET_TENSORRT "Enable exporting TensorRT dialect IR to a TensorRT engine" ON)
option(MLIR_TRT_TARGET_CPP "Enable exporting TensorRT dialect IR to C++ code that calls the TensorRT API" ON)
option(MLIR_TRT_ENABLE_PYTHON "Enable building the mlir_tensorrt_compiler python package." ON)
option(MLIR_TRT_ENABLE_NVTX "Enable NVTX tracing" ON)
option(MLIR_TRT_ENABLE_TESTING "Enable building optional tests" ON)
option(MLIR_TRT_TARGET_LUA "Enable translating MLIR to the Lua target" ON)
option(MLIR_TRT_ENABLE_EXECUTOR "Build the Executor dialect and MLIR-TensorRT Execution Engine" ON)
option(MLIR_TRT_ENABLE_NCCL "Enable the NCCL runtime module" ON)

set(MLIR_TRT_TENSORRT_DIR "" CACHE STRING "Path to TensorRT install directory")
set(MLIR_TRT_DOWNLOAD_TENSORRT_VERSION "10.0" CACHE STRING
   "Version of TensorRT to download and use. It overrides MLIR_TRT_TENSORRT_DIR.")
set(MLIR_TRT_PACKAGE_CACHE_DIR "" CACHE STRING "Directory where to cache downloaded C++ packages")
set(MLIR_TRT_USE_LINKER "" CACHE STRING "Specify a linker to use (e.g. LLD); this is just an alias for LLVM_USE_LINKER")

# -------------------------------------------------
# Option validation
# -------------------------------------------------

if(MLIR_TRT_PACKAGE_CACHE_DIR)
  set(CPM_SOURCE_CACHE "${MLIR_TRT_PACKAGE_CACHE_DIR}" CACHE STRING "" FORCE)
  set(CPM_USE_NAMED_CACHE_DIRECTORIES ON CACHE BOOL "" FORCE)
else()
  message(WARNING "MLIR_TRT_PACKAGE_CACHE_DIR is not set. Downloaded packages will be \
  stored in your build directory. It is highly recommended to specify a package cache directory outside of \
  your build directory (for example '$PWD/.cache.cpm')")
endif()

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

# -------------------------------------------------
# Include utils/tools
# -------------------------------------------------
include(build_tools/cmake/CPM.cmake)
include(build_tools/cmake/CompilationOptions.cmake)
include(build_tools/cmake/Targets.cmake)
include(build_tools/cmake/Dependencies.cmake)


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

if(MLIR_TRT_ENABLE_HLO)
  mtrt_add_stablehlo(
    # We don't tag versions vX.Y.Z in stablehlo, so use the upgrade
    # year-month-day as a tag for CPM caching.
    VERSION 0.0.20240624
    GIT_TAG fb18ee251b81a43fd9a0f45f2f34770389e916d0
    GIT_REPOSITORY "https://github.com/openxla/stablehlo.git"
    PATCHES "${MLIR_TENSORRT_ROOT_DIR}/build_tools/stablehlo.patch"
    OPTIONS
      "STABLEHLO_ENABLE_BINDINGS_PYTHON ${MLIR_TRT_ENABLE_PYTHON}"
      "STABLEHLO_BUILD_EMBEDDED ON"
    EXCLUDE_FROM_ALL TRUE
  )
endif()

if(MLIR_TRT_ENABLE_TESTING)
  # TODO: force the flag that makes LLVM build google benchmark.
endif()


if(MLIR_TRT_TARGET_TENSORRT)
  find_tensorrt(
    INSTALL_DIR "${MLIR_TRT_TENSORRT_DIR}"
    DOWNLOAD_VERSION "${MLIR_TRT_DOWNLOAD_TENSORRT_VERSION}"
    MIN_VERSION "${min_tensorrt_version}"
  )
endif()

if(MLIR_TRT_ENABLE_PYTHON)
  mlir_tensorrt_find_dlpack()
endif()

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

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_compile_options(-Wno-deprecated-declarations)

include(AddMLIRPython)
include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

# Add a meta target for all documentation generation targets. You can generate
# all documentation under `${buildDir}/docs` by building this target.
add_custom_target("mlir-tensorrt-doc")

if(MLIR_TRT_ENABLE_EXECUTOR)
  add_subdirectory(executor EXCLUDE_FROM_ALL)
endif()

add_subdirectory(tensorrt EXCLUDE_FROM_ALL)
include_directories(${CMAKE_CURRENT_LIST_DIR}/tensorrt/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/tensorrt/include)

add_subdirectory(compiler)
add_subdirectory(python)

if(MLIR_TRT_ENABLE_TESTING)
  add_subdirectory(test)
endif()

add_subdirectory(tools)
