#-------------------------------------------------------------------------------------
# Creates a MLIR-Kernel library with proper setup and installation
#
# Parameters:
#   name - Target name (required)
#   All other arguments are passed to mtrt_add_project_library
#
# Usage:
#   add_mlir_kernel_library(MyKernelLib SOURCES file1.cpp file2.cpp)
#   add_mlir_kernel_library(MyKernelLib DISABLE_INSTALL SOURCES file1.cpp)
#-------------------------------------------------------------------------------------
function(add_mlir_kernel_library name)
  mtrt_add_project_library(${name}
    PROJECT_NAME Kernel
    ${ARGN}
  )
endfunction()

#-------------------------------------------------------------------------------------
# Locates the CUDA libdevice.10.bc file
#
# Parameters:
#   outputVar - Variable name to store the libdevice path (required)
#
# Usage:
#   mtrt_find_libdevice_file(LIBDEVICE_PATH)
#
# Note: Sets the variable in the parent scope with the path to libdevice.10.bc
# TODO: This should change to embed libdevice
#-------------------------------------------------------------------------------------
function(mtrt_find_libdevice_file outputVar)
  # Validate required arguments
  if(NOT outputVar)
    message(FATAL_ERROR "mtrt_find_libdevice_file: outputVar parameter is required")
  endif()

  find_package(CUDAToolkit REQUIRED)

  find_file(CudaLibDevicePath NAMES libdevice.10.bc
    HINTS ${CUDAToolkit_LIBRARY_ROOT}
    PATHS ${CUDAToolkit_LIBRARY_ROOT}
    PATH_SUFFIXES nvvm libdevice nvvm/libdevice
    REQUIRED
  )

  if(NOT CudaLibDevicePath)
    message(FATAL_ERROR "Could not locate libdevice.10.bc in CUDA toolkit at ${CUDAToolkit_LIBRARY_ROOT}")
  endif()

  message(STATUS "Found libdevice at: ${CudaLibDevicePath}")
  set("${outputVar}" "${CudaLibDevicePath}" PARENT_SCOPE)
endfunction()


#-------------------------------------------------------------------------------------
# Creates a MLIR-Kernel interface library with TableGen generation
#
# Parameters:
#   target - Target name (required)
#   TD - TableGen definition file path (required)
#   OP - Operation interface output path (optional)
#   ATTR - Attribute interface (optional)
#   All other arguments are passed to add_mlir_kernel_library
#
# Usage:
#   add_mlir_kernel_interface_library(MyInterface TD "path/to/interface.td" OP "MyInterface")
#-------------------------------------------------------------------------------------
function(add_mlir_kernel_interface_library target)
  # Validate required arguments
  if(NOT target)
    message(FATAL_ERROR "add_mlir_kernel_interface_library: target parameter is required")
  endif()

  cmake_parse_arguments(ARG "" "ATTR;OP;TD" "" ${ARGN})

  if(NOT ARG_TD)
    message(FATAL_ERROR "add_mlir_kernel_interface_library: TD parameter is required")
  endif()

  cmake_path(SET ARG_TD
    NORMALIZE
    "${PROJECT_SOURCE_DIR}/include/${ARG_TD}")

  set(LLVM_TARGET_DEFINITIONS "${ARG_TD}")

  if(ARG_OP)
    cmake_path(SET ARG_OP
      NORMALIZE
      "${PROJECT_BINARY_DIR}/include/${ARG_OP}")
    cmake_path(RELATIVE_PATH ARG_OP
        BASE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    mlir_tablegen("${ARG_OP}.h.inc" -gen-op-interface-decls)
    mlir_tablegen("${ARG_OP}.cpp.inc" -gen-op-interface-defs)
  endif()

  mtrt_add_public_tablegen_target(${target}IncGen)

  add_mlir_kernel_library(${target}
    PARTIAL_SOURCES_INTENDED
    ${ARG_UNPARSED_ARGUMENTS}
    DEPENDS ${target}IncGen)
endfunction()
