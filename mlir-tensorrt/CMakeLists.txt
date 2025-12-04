cmake_minimum_required(VERSION 3.25)
project(mlir-tensorrt LANGUAGES CXX C)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/build_tools/cmake")

include(MTRTCMakePolicy NO_POLICY_SCOPE)

include(CMakeDependentOption)
include(CMakePrintHelpers)
include(MTRTCPM)
include(MTRTCMakeExtras)
include(MTRTDependencies)
include(MTRTFeatures)
include(AddMLIRTensorRT)
include(Version.cmake)
include(CMakeOptions.cmake)

# -------------------------------------------------
# Package cache configuration
# -------------------------------------------------
if(NOT CPM_SOURCE_CACHE)
  message(WARNING "CPM_SOURCE_CACHE is not set. Source code for third party C++ packages will be \
  stored in your build directory. It is highly recommended to specify a CPM source cache directory outside of \
  your build directory (for example '$PWD/.cache.cpm')")
endif()


# List of dialects to embed in the Python package. This is different between internal
# and public versions of the code.
set(MLIR_TRT_PYTHON_UPSTREAM_DIALECTS_EMBED
  affine arith bufferization builtin complex func math quant scf
  tensor
  )

# List of dialects to embed in the Python package. This is different between internal
# and public versions of the code.
set(MLIR_TRT_PYTHON_UPSTREAM_DIALECTS_EMBED
  affine arith bufferization builtin complex index func gpu math memref quant scf
  tensor linalg
  )

# Specifies whether we require CUDA at build time.
set(MLIR_TRT_REQUIRES_CUDA FALSE)
if(MLIR_TRT_TARGET_TENSORRT OR
   MLIR_TRT_ENABLE_CUBLAS OR
   MLIR_TRT_ENABLE_NCCL)
  set(MLIR_TRT_REQUIRES_CUDA TRUE)
endif()

if(MLIR_TRT_REQUIRES_CUDA AND NOT MLIR_TRT_ENABLE_CUDA)
  message(FATAL_ERROR "CUDA is required but MLIR_TRT_ENABLE_CUDA is disabled")
endif()

# -------------------------------------------------
# Option validation
# -------------------------------------------------

# Setup use of ccache for C, CXX, and CUDA languages if we are the top-level
# project.
# Enables generation of compile_commands.json for clangd.
if(PROJECT_IS_TOP_LEVEL)
  include(MTRTCompilerCache)
  mtrt_use_compiler_cache()
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

# -------------------------------------------------
# Setup LLVM/MLIR
# -------------------------------------------------
find_package(Python3 REQUIRED
  COMPONENTS Interpreter Development)

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

# Further modify global compilation flags. This should
# appear after `HandleLLVMOptions`.
include(MTRTCompilationOptions)

# -------------------------------------------------
# Dependencies
# -------------------------------------------------

find_package(NVTX REQUIRED)
find_package(DLPack REQUIRED)

if(MLIR_TRT_REQUIRES_CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

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

if(MLIR_TRT_ENABLE_MPI)
  find_package(MPI COMPONENTS C)
endif()

include(AddMLIRPython)
include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

find_package(TVMFFI REQUIRED)

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


# Add a custom target for header generation. Since this target is used for
# forcing generation of all generated headers prior to generation of
# a distribution, also make it depend on mlir-headers.
add_custom_target(mtrt-headers)
if(TARGET mlir-headers)
  add_dependencies(mtrt-headers mlir-headers)
endif()

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

mtrt_write_feature_flags_header()
include_directories("${CMAKE_CURRENT_BINARY_DIR}/include")

add_compile_options(-Wno-deprecated-declarations)

# Add a meta target for all documentation generation targets. You can generate
# all documentation under `${buildDir}/docs` by building this target.
add_custom_target("mlir-tensorrt-doc")

# Add a meta target for running all sub-project `check-` targets.
add_custom_target(check-all-mlir-tensorrt)
add_custom_target(check-all-mlir-tensorrt-build-only)

add_subdirectory(common)
foreach(project_name IN LISTS MLIR_TRT_ENABLE_PROJECTS)
  add_subdirectory(${project_name})
endforeach()
