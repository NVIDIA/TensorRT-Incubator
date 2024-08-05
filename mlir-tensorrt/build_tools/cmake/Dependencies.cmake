include(CMakeParseArguments)

#-------------------------------------------------------------------------------------
# Wrapper around CPMAddPackage
#-------------------------------------------------------------------------------------
macro(mlir_tensorrt_add_package)
  CPMAddPackage(
    ${ARGN}
  )
endmacro()

# ------------------------------------------------------------------------------
# Downloads the Google Benchmark C++ library and adds it to the build.
# ------------------------------------------------------------------------------
function(mtrt_add_google_benchmark)
  CPMAddPackage(
    NAME benchmark  GITHUB_REPOSITORY google/benchmark
    VERSION 1.8.3
    EXCLUDE_FROM_ALL TRUE
    GIT_SHALLOW TRUE
    OPTIONS
    "BENCHMARK_ENABLE_TESTING OFF"
    "BENCHMARK_USE_BUNDLED_GTEST OFF"
    "BENCHMARK_ENABLE_WERROR OFF"
    "BENCHMARK_ENABLE_GTEST_TESTS OFF"
  )
endfunction()

#-------------------------------------------------------------------------------------
# Downloads stablehlo and adds it to the build.
# TODO: currently the backup commit hash for external release needs to be manually
# updated for each release.
#-------------------------------------------------------------------------------------
function(mtrt_add_stablehlo)
  CPMAddPackage(
        NAME stablehlo
        ${ARGN}
  )
  set(stablehlo_SOURCE_DIR "${stablehlo_SOURCE_DIR}" PARENT_SCOPE)
endfunction()

#-------------------------------------------------------------------------------------
# Given a Path to `NvInfer.h`, extracts the TensorRT version and checks against
# expected version.
#-------------------------------------------------------------------------------------
macro(get_tensorrt_version nvinfer_version_file out_var)
  file(STRINGS "${nvinfer_version_file}" VERSION_STRINGS REGEX "#define NV_TENSORRT_.*")
  foreach(TYPE MAJOR MINOR PATCH BUILD)
    string(REGEX MATCH "NV_TENSORRT_${TYPE} [0-9]+" TRT_TYPE_STRING ${VERSION_STRINGS})
    string(REGEX MATCH "[0-9]+" TRT_${TYPE} ${TRT_TYPE_STRING})
  endforeach(TYPE)
  set("${out_var}" "${TRT_MAJOR}.${TRT_MINOR}.${TRT_PATCH}.${TRT_BUILD}")
endmacro()

# -------------------------------------------------------------------------------------
# Downloads TensorRT given specified version.
# Stores extracted path in variable given to `OUT_VAR`.
# Usage: `download_tensorrt(VERSION 9.1.0.4 OUT_VAR TENSORRT_DOWNLOAD_DIR)
# -------------------------------------------------------------------------------------
function(download_tensorrt)
  cmake_parse_arguments(ARG "" "VERSION;OUT_VAR" "" ${ARGN})

  if((NOT ARG_VERSION) OR (NOT ARG_OUT_VAR))
    message(FATAL_ERROR "Expected VERSION, OUT_VAR arguments to download_tensorrt(...)")
  endif()

  if(MSVC)
    message(FATAL_ERROR "Windows automatic download of TensorRT is unsupported")
  endif()

  if((NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    OR(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64"))
    message(FATAL_ERROR "automatic download of TensorRT is only supported on x86_64 Linux platforms")
  endif()

  # Canonicalize "10.0" version by setting it to the latest public TRT 10.0 version.
  if(ARG_VERSION VERSION_EQUAL "10.0")
    set(ARG_VERSION "10.0.0.6")
  endif()

  # Canonicalize "10.1" version by setting it to the latest public TRT 10.1 version.
  if(ARG_VERSION VERSION_EQUAL "10.1")
    set(ARG_VERSION "10.1.0.27")
  endif()
  # Canonicalize "10.2" version by setting it to the latest public TRT 10.2 version.
  if(ARG_VERSION VERSION_EQUAL "10.2")
    set(ARG_VERSION "10.2.0.19")
  endif()

  set(downloadable_versions
    "9.0.1.4" "9.1.0.4" "9.2.0.5"
    "10.0.0.6" "10.1.0.27"
    "10.2.0.19"
  )

  if(NOT ARG_VERSION IN_LIST downloadable_versions)
    message(FATAL_ERROR "CMake download of TensorRT is only available for \
      the following versions: ${downloadable_versions}")
  endif()

  set(TRT_VERSION "${ARG_VERSION}")

  # Handle TensorRT 9 versions. These are publicly accessible download links.
  if(ARG_VERSION VERSION_LESS 10.0.0 AND ARG_VERSION VERSION_GREATER 9.0.0)
    string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" trt_short_version ${ARG_VERSION})
    set(CUDA_VERSION "12.2")
    set(OS "linux")
    EXECUTE_PROCESS(COMMAND uname -m
                    COMMAND tr -d '\n'
                    OUTPUT_VARIABLE ARCH)
    if(ARCH STREQUAL "arm64")
      set(ARCH "aarch64")
      set(OS "ubuntu-20.04")
    elseif(ARCH STREQUAL "amd64")
      set(ARCH "x86_64")
    elseif(ARCH STREQUAL "aarch64")
      set(OS "ubuntu-20.04")
    elseif(NOT (ARCH STREQUAL "x86_64"))
      message(FATAL_ERROR "Direct download not available for architecture: ${ARCH}")
    endif()
    if(ARG_VERSION VERSION_LESS 9.2.0)
      set(_url "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/${trt_short_version}/tars/tensorrt-${TRT_VERSION}.${OS}.${ARCH}-gnu.cuda-${CUDA_VERSION}.tar.gz")
    else()
      set(_url "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/${trt_short_version}/tensorrt-${TRT_VERSION}.${OS}.${ARCH}-gnu.cuda-${CUDA_VERSION}.tar.gz")
    endif()
  endif()

  if(ARG_VERSION VERSION_EQUAL 10.0.0.6)
    set(_url "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.0/tensorrt-10.0.0.6.linux.x86_64-gnu.cuda-12.4.tar.gz")
  endif()

  if(ARG_VERSION VERSION_EQUAL 10.1.0.27)
    set(_url "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.1.0/tars/tensorrt-10.1.0.27.linux.x86_64-gnu.cuda-12.4.tar.gz")
  endif()

  if(ARG_VERSION VERSION_EQUAL 10.2.0.19)
    set(_url "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.2.0/tars/TensorRT-10.2.0.19.Linux.x86_64-gnu.cuda-12.5.tar.gz")
  endif()

  if(NOT _url)
    message(FATAL_ERROR "Could not determine TensorRT download URL")
  endif()

  message(STATUS "TensorRT Download URL: ${_url}")

  CPMAddPackage(
    NAME TensorRT9
    VERSION "${TRT_VERSION}"
    URL ${_url}
    DOWNLOAD_ONLY
  )
  set("${ARG_OUT_VAR}" "${TensorRT9_SOURCE_DIR}" PARENT_SCOPE)
endfunction()

#-------------------------------------------------------------------------------------
# Finds the TensorRT headers and creates an interface target `TensorRTHeaders`.
# Usage:
# `find_tensorrt(INSTALL_DIR "/path/to/TensorRT-X.Y.Z.W" MIN_VERSION 1.2.3)`
#-------------------------------------------------------------------------------------
function(find_tensorrt)
  cmake_parse_arguments(ARG "" "INSTALL_DIR;MIN_VERSION;DOWNLOAD_VERSION" "" ${ARGN})

  if(TARGET TensorRTHeaderOnly)
    return()
  endif()

  # If use specified a version to download, try to do that first.
  if(ARG_DOWNLOAD_VERSION)
    download_tensorrt(
      VERSION ${ARG_DOWNLOAD_VERSION}
      OUT_VAR TensorRT_SOURCE_DIR
    )
    set(_tensorrt_include_dir "${TensorRT_SOURCE_DIR}/include")
    # Force override MLIR_TRT_TENSORRT_DIR.
    set(MLIR_TRT_TENSORRT_DIR CACHE STRING "${TensorRT_SOURCE_DIR}" FORCE)
    set(ARG_INSTALL_DIR "${TensorRT_SOURCE_DIR}")
  endif()

  if(ARG_INSTALL_DIR)
    message(STATUS "Looking for TensorRT headers in ${ARG_INSTALL_DIR}")
    find_path(
      _tensorrt_include_dir
      NAMES NvInfer.h
      HINTS ${ARG_INSTALL_DIR} ${ARG_INSTALL_DIR}/include
      PATHS ${ARG_INSTALL_DIR} ${ARG_INSTALL_DIR}/include
      REQUIRED
      NO_CMAKE_PATH NO_DEFAULT_PATH
      NO_CACHE
      )
    find_path(
      _tensorrt_lib_dir
      NAMES libnvinfer.so
      HINTS ${ARG_INSTALL_DIR} ${ARG_INSTALL_DIR}/lib
      PATHS ${ARG_INSTALL_DIR} ${ARG_INSTALL_DIR}/lib
      NO_CMAKE_PATH NO_DEFAULT_PATH
      NO_CACHE
    )
  else()
    message(WARNING
    "MLIR_TRT_TENSORRT_DIR not set, looking for NvInfer.h in system include dirs")
    find_path(
      _tensorrt_include_dir
      NAMES NvInfer.h
      REQUIRED
      NO_CACHE
    )
    find_library(
      _tensorrt_lib
      NAMES libnvinfer.so
      NO_CACHE
    )
    if(_tensorrt_lib)
      get_filename_component(_tensorrt_lib_dir "${_tensorrt_lib}" DIRECTORY)
    endif()
  endif()

  if(NOT _tensorrt_lib_dir)
    message(WARNING "Could not locate libnvinfer.so it is not required for building, \
    but path to nvinfer.so must be given via LD_LIBRARY_PATH at runtime.")
  endif()

  get_tensorrt_version("${_tensorrt_include_dir}/NvInferVersion.h" TRT_VERSION)
  message(STATUS "Found TensorRT version: ${TRT_VERSION} at ${_tensorrt_include_dir}")

  if(TRT_VERSION VERSION_LESS ARG_MIN_VERSION)
    message(FATAL_ERROR "Found TensorRT Version ${TRT_VERSION}, but version at least ${ARG_MIN_VERSION} is required")
  endif()

  set(MLIR_TRT_TENSORRT_LIB_DIR "${_tensorrt_lib_dir}" PARENT_SCOPE)
  set(MLIR_TRT_TENSORRT_VERSION "${TRT_VERSION}" PARENT_SCOPE)

  add_library(TensorRTHeaderOnly INTERFACE IMPORTED)
  target_include_directories(TensorRTHeaderOnly INTERFACE
    $<BUILD_INTERFACE:${_tensorrt_include_dir}>
    )
  target_compile_options(TensorRTHeaderOnly INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
    )
endfunction()

#-------------------------------------------------------------------------------------
# Download and add DLPack to the build (header only)
#-------------------------------------------------------------------------------------
function(mlir_tensorrt_find_dlpack)
  CPMAddPackage(
    NAME dlpack
    VERSION 1.0rc
    URL https://github.com/dmlc/dlpack/archive/refs/tags/v1.0rc.tar.gz
    DOWNLOAD_ONLY TRUE
  )
  if(NOT TARGET DLPackHeaderOnly)
    add_library(DLPackHeaderOnly INTERFACE IMPORTED)
    target_include_directories(DLPackHeaderOnly INTERFACE
      $<BUILD_INTERFACE:${dlpack_SOURCE_DIR}/include>)
    add_library(DLPack::Headers ALIAS DLPackHeaderOnly)
  endif()
endfunction()

