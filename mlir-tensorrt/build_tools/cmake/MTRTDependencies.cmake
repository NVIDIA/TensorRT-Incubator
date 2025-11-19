include(CMakeParseArguments)
include(${CMAKE_CURRENT_LIST_DIR}/TensorRTDownloadURL.cmake)

#-------------------------------------------------------------------------------------
# Given a Path to `NvInfer.h`, extracts the TensorRT version and checks against
# expected version.
#-------------------------------------------------------------------------------------
macro(get_tensorrt_version nvinfer_version_file out_var)
  file(STRINGS "${nvinfer_version_file}" VERSION_STRINGS REGEX "#define (TRT_.+|NV_TENSORRT_.+) [0-9]+")
  foreach(TYPE MAJOR MINOR PATCH BUILD)
    string(REGEX MATCH "(TRT_${TYPE}_ENTERPRISE|NV_TENSORRT_${TYPE}) [0-9]+" TRT_TYPE_STRING ${VERSION_STRINGS})
    if("${TRT_TYPE_STRING}" STREQUAL "")
      message(FATAL_ERROR "Failed to extract TensorRT ${TYPE} version from ${nvinfer_version_file}")
    endif()
    string(REGEX MATCH "[0-9]+" "TRT_${TYPE}" "${TRT_TYPE_STRING}")
    if("TRT_${TYPE}" STREQUAL "")
      message(FATAL_ERROR "Failed to extract TensorRT ${TYPE} version from ${nvinfer_version_file}")
    endif()
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

  mtrt_get_tensorrt_download_url(
    "${ARG_VERSION}"
    "${CMAKE_SYSTEM_NAME}"
    "${CMAKE_SYSTEM_PROCESSOR}"
    _url
    _updated_version
  )

  message(STATUS "TensorRT Download URL: ${_url}")

  mlir_tensorrt_add_package(
    NAME TensorRT
    VERSION "${ARG_VERSION}"
    URL ${_url}
    CUSTOM_CACHE_KEY "${_updated_version}-${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_SYSTEM_NAME}"
    DOWNLOAD_ONLY
  )
  set("${ARG_OUT_VAR}" "${TensorRT_SOURCE_DIR}" PARENT_SCOPE)
endfunction()

macro(configure_tensorrt_python_plugin_header)
  if(ARG_INSTALL_DIR)
    find_file(
      trt_python_plugin_header
      NAMES NvInferPythonPlugin.h plugin.h
      HINTS ${ARG_INSTALL_DIR} ${ARG_INSTALL_DIR}/python/include/impl
      PATHS ${ARG_INSTALL_DIR} ${ARG_INSTALL_DIR}/python/include/impl
      REQUIRED
      NO_CMAKE_PATH NO_DEFAULT_PATH
      NO_CACHE
    )
  else()
    find_path(
      trt_python_plugin_header
      NAMES NvInferPythonPlugin.h plugin.h
      REQUIRED
      NO_CACHE
    )
  endif()
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include/nvinfer")
  file(COPY_FILE "${trt_python_plugin_header}"
    "${CMAKE_BINARY_DIR}/include/nvinfer/trt_plugin_python.h"
    ONLY_IF_DIFFERENT
    RESULT copy_result
  )
  if(copy_result)
    message(FATAL_ERROR "failed to copy TensorRT QDP plugin header: ${copy_result}")
  endif()
endmacro()

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

  if(TRT_VERSION VERSION_GREATER_EQUAL 10.9)
    add_compile_definitions(ENABLE_AOT_PLUGIN=1)
    configure_tensorrt_python_plugin_header()
    message(STATUS "Found TensorRT Python headers at ${trt_python_plugin_header}")
  endif()

  set(MLIR_TRT_TENSORRT_LIB_DIR "${_tensorrt_lib_dir}" PARENT_SCOPE)
  set(MLIR_TRT_TENSORRT_VERSION "${TRT_VERSION}" PARENT_SCOPE)

  add_library(TensorRTHeaderOnly INTERFACE IMPORTED GLOBAL)
  target_include_directories(TensorRTHeaderOnly INTERFACE
    $<BUILD_INTERFACE:${_tensorrt_include_dir}>
    )
  if(TRT_VERSION VERSION_GREATER_EQUAL 10.9)
    target_include_directories(TensorRTHeaderOnly INTERFACE
      $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
      )
  endif()
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
