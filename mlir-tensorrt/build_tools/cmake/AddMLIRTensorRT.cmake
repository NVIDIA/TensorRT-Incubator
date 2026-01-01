#-------------------------------------------------------------------------------------
# MLIR-TensorRT CMake utility functions
#
# This file provides CMake functions for building MLIR-TensorRT libraries,
# dialects, and related targets. All functions follow modern CMake best practices
# with proper parameter validation and documentation.
#-------------------------------------------------------------------------------------
include(MTRTCommonFunctions)

#-------------------------------------------------------------------------------------
# Creates a MLIR-TensorRT library with proper installation and global registration
#
# Parameters:
#   target - Target name (required)
#   All other arguments are passed to add_mlir_library
#
# Usage:
#   add_mlir_tensorrt_library(MyLibrary SOURCES file1.cpp file2.cpp)
#-------------------------------------------------------------------------------------
function(add_mlir_tensorrt_library target)
  mtrt_add_project_library(${target}
    PROJECT_NAME MLIRTensorRT
    ${ARGN}
  )
endfunction()

#-------------------------------------------------------------------------------------
# Creates a MLIR-TensorRT dialect library with global registration
#
# Parameters:
#   target - Target name (required)
#   All other arguments are passed to add_mlir_dialect_library
#
# Usage:
#   add_mlir_tensorrt_dialect_library(MyDialect SOURCES dialect.cpp)
#-------------------------------------------------------------------------------------
function(add_mlir_tensorrt_dialect_library target)
  # Validate required arguments
  if(NOT target)
    message(FATAL_ERROR "add_mlir_tensorrt_dialect_library: target parameter is required")
  endif()

  mtrt_add_project_library(${target}
    PROJECT_NAME MLIRTensorRT
    LIBRARY_TYPE DIALECT
    ${ARGN}
  )
endfunction()

#-------------------------------------------------------------------------------------
# Declares an operation interface using TableGen
#
# Parameters:
#   name - Interface name (required)
#
# Usage:
#   add_mlir_tensorrt_op_interface(MyInterface)
#
# Note: Expects a file named ${name}.td in the current directory
#-------------------------------------------------------------------------------------
function(add_mlir_tensorrt_op_interface name)
  # Validate required arguments
  if(NOT name)
    message(FATAL_ERROR "add_mlir_tensorrt_op_interface: name parameter is required")
  endif()

  set(LLVM_TARGET_DEFINITIONS "${name}.td")
  mlir_tablegen(${name}.h.inc -gen-op-interface-decls)
  mlir_tablegen(${name}.cpp.inc -gen-op-interface-defs)
  mtrt_add_public_tablegen_target("MLIRTensorRT${name}IncGen")
endfunction()

#-------------------------------------------------------------------------------------
# Declares an attribute interface using TableGen
#
# Parameters:
#   src - Source file name without .td extension (required)
#   OUTPUT_NAME - Custom output name (optional, defaults to src)
#
# Usage:
#   add_mlir_tensorrt_attr_interface(MyAttr OUTPUT_NAME CustomName)
#-------------------------------------------------------------------------------------
function(add_mlir_tensorrt_attr_interface src)
  # Validate required arguments
  if(NOT src)
    message(FATAL_ERROR "add_mlir_tensorrt_attr_interface: src parameter is required")
  endif()

  cmake_parse_arguments(ARG "" "OUTPUT_NAME" "" ${ARGN})
  if(NOT ARG_OUTPUT_NAME)
    set(ARG_OUTPUT_NAME "${src}")
  endif()

  set(LLVM_TARGET_DEFINITIONS "${src}.td")
  mlir_tablegen("${ARG_OUTPUT_NAME}.h.inc" -gen-attr-interface-decls)
  mlir_tablegen("${ARG_OUTPUT_NAME}.cpp.inc" -gen-attr-interface-defs)
  mtrt_add_public_tablegen_target("MLIRTensorRT${ARG_OUTPUT_NAME}IncGen")
endfunction()

#-------------------------------------------------------------------------------------
# Parses a library list with PUBLIC/PRIVATE keywords
#
# Parameters:
#   out_public - Variable to store PUBLIC libraries (required)
#   out_private - Variable to store PRIVATE libraries (required)
#   PUBLIC - List of public libraries
#   PRIVATE - List of private libraries
#
# Usage:
#   mlir_tensorrt_parse_library_list(PUB_LIBS PRIV_LIBS
#     PUBLIC lib1 lib2 PRIVATE lib3 lib4)
#-------------------------------------------------------------------------------------
function(mlir_tensorrt_parse_library_list out_public out_private)
  # Validate required arguments
  if(NOT out_public OR NOT out_private)
    message(FATAL_ERROR "mlir_tensorrt_parse_library_list: out_public and out_private parameters are required")
  endif()

  cmake_parse_arguments(ARG "" "" "PUBLIC;PRIVATE" ${ARGN})
  set("${out_public}" ${ARG_PUBLIC} PARENT_SCOPE)
  set("${out_private}" ${ARG_PRIVATE} PARENT_SCOPE)
endfunction()

# --------------------------------------------------------------
# Creates `target` that invokes the `flatc` compiler on the
# given SRC, which sould be a flatbuffer schema file.
# It creates target which at build time generates a corresponding
# [filename]Generated.h in the build directory corresponding to the
# source directory of the SRC file.
# --------------------------------------------------------------
function(add_mlir_tensorrt_flatbuffer_schema target)
  set(prefix ARG)
  set(noValues "")
  set(singleValues "SRC")
  cmake_parse_arguments(${prefix} "${noValues}" "${singleValues}"
                        "${multiValues}" ${ARGN})
  get_filename_component(srcFileName "${ARG_SRC}" NAME_WE)
  set(generatedFileName "${srcFileName}Flatbuffer.h")
  add_custom_command(
    OUTPUT "${generatedFileName}"
    COMMAND flatc --cpp --cpp-std c++17 -o ${CMAKE_CURRENT_BINARY_DIR}
            --filename-suffix Flatbuffer "${ARG_SRC}"
            --gen-object-api
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SRC}"
  )
  add_custom_target(${target}
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${generatedFileName}"
  )
endfunction()

# --------------------------------------------------------------
# Creates `targetName` that invokes mlir-tensorrt-tblgen
# on a [dialect]Ops.td file to generate implementations for the
# TensorRTEncodingOpInterface's encodeOp interface method.
# --------------------------------------------------------------
function(add_tensorrt_encoding_def_gen targetName inputFileName outputFileName )
  list(TRANSFORM MLIR_INCLUDE_DIRS PREPEND "-I" OUTPUT_VARIABLE _mlir_includes)
  add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${outputFileName}"
  COMMAND mlir-tensorrt-tblgen --gen-tensorrt-layer-add-defs
    "${inputFileName}"
    -I "${MLIR_TENSORRT_ROOT_DIR}/tensorrt/include"
    ${_mlir_includes}
    -o "${CMAKE_CURRENT_BINARY_DIR}/${outputFileName}"
  DEPENDS "${inputFileName}" mlir-tensorrt-tblgen
  )
  add_custom_target(${targetName} DEPENDS
    "${CMAKE_CURRENT_BINARY_DIR}/${outputFileName}")
endfunction()

# ------------------------------------------------------------------------------
# Reads given file at specified path and captures line with given regex.
# The `${out_var}` is populated with the result of ${capture_group} of the regex.
# ------------------------------------------------------------------------------
function(_mtrt_find_in_file filename regex capture_group out_var)
  # Get the expected version number for the package.
  file(STRINGS ${filename} _tmp
    REGEX ${regex}
  )
  string(REGEX REPLACE ${regex} ${capture_group} "${out_var}" "${_tmp}")
  return(PROPAGATE "${out_var}")
endfunction()

# ------------------------------------------------------------------------------
# Generate markdown documentation.
# ------------------------------------------------------------------------------
function(add_mlir_tensorrt_doc name)
  if(NOT TARGET mlir-tensorrt-doc)
    add_custom_target(mlir-tensorrt-doc)
  endif()
  cmake_parse_arguments(ARG "" "SRC;OUTPUT_FILE" "COMMAND" ${ARGN})
  set(LLVM_TARGET_DEFINITIONS "${ARG_SRC}")
  tablegen(MLIR "${name}.md" ${ARG_COMMAND})
  add_custom_command(
          OUTPUT "${CMAKE_BINARY_DIR}/${ARG_OUTPUT_FILE}"
          COMMAND ${CMAKE_COMMAND} -E copy
                  "${name}.md"
                  "${CMAKE_BINARY_DIR}/${ARG_OUTPUT_FILE}"
          DEPENDS "${name}.md")
  add_custom_target("MLIRTensorRT${name}DocGen" DEPENDS "${CMAKE_BINARY_DIR}/${ARG_OUTPUT_FILE}")
  add_dependencies("mlir-tensorrt-doc" "MLIRTensorRT${name}DocGen")
endfunction()

# ------------------------------------------------------------------------------
# Declare a Plan dialect extension backend library.
# ------------------------------------------------------------------------------
function(add_mlir_tensorrt_backend_library target)
  cmake_parse_arguments(ARG "" "TD" "" ${ARGN})
  cmake_path(SET SRC_TD
    NORMALIZE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../include/${ARG_TD}")
  cmake_path(SET BIN_TD
    NORMALIZE
    "${CMAKE_CURRENT_BINARY_DIR}/../../../include/${ARG_TD}")
  # Tablegen output paths have to be relative to CMAKE_CURRENT_BINARY_DIR.
  cmake_path(RELATIVE_PATH BIN_TD
    BASE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  # Check for Passes.td in the same directory as the backend TD file
  cmake_path(GET SRC_TD PARENT_PATH backend_dir)
  cmake_path(SET passes_td_path "${backend_dir}/Passes.td")

  set(tablegen_depends "")

  # Generate Passes.h.inc if Passes.td exists and target doesn't already exist
  if(EXISTS "${passes_td_path}")
    set(passes_tablegen_target "${target}PassesIncGen")

    # Only create the target if it doesn't already exist (e.g., for internal backends)
    if(NOT TARGET "${passes_tablegen_target}")
      # Extract backend name from TD path (e.g., "Host" from "Backends/Host/HostBackend.td")
      cmake_path(GET backend_dir FILENAME backend_name)

      # Get the relative path from the include directory
      # backend_dir is something like /workspaces/mlir-tensorrt/compiler/include/mlir-tensorrt/Backends/Host
      # We need mlir-tensorrt/Backends/Host
      cmake_path(SET include_dir
        NORMALIZE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../include")
      file(RELATIVE_PATH backend_dir_rel
        "${include_dir}"
        "${backend_dir}")

      # Set up Passes.td tablegen - output should be relative to CMAKE_CURRENT_BINARY_DIR
      cmake_path(SET BIN_PASSES_TD
        NORMALIZE
        "${CMAKE_CURRENT_BINARY_DIR}/../../../include/${backend_dir_rel}/Passes.td")
      cmake_path(RELATIVE_PATH BIN_PASSES_TD
        BASE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

      set(LLVM_TARGET_DEFINITIONS "${passes_td_path}")
      string(REPLACE "Passes.td" "Passes.h.inc" passes_h_inc ${BIN_PASSES_TD})

      mlir_tablegen("${passes_h_inc}" -gen-pass-decls -name ${backend_name}Backend)

      mtrt_add_public_tablegen_target("${passes_tablegen_target}")
    endif()

    list(APPEND tablegen_depends "${passes_tablegen_target}")
  endif()

  # Generate backend attribute files
  set(LLVM_TARGET_DEFINITIONS "${SRC_TD}")

  string(REPLACE ".td" "Attrs.h.inc" h_inc_file ${BIN_TD})
  string(REPLACE ".td" "Attrs.cpp.inc" cpp_inc_file ${BIN_TD})
  mlir_tablegen("${h_inc_file}" -gen-attrdef-decls)
  mlir_tablegen("${cpp_inc_file}" -gen-attrdef-defs)

  mtrt_add_public_tablegen_target("${target}IncGen")
  list(APPEND tablegen_depends "${target}IncGen")

  add_dependencies("${target}IncGen"
    MLIRTensorRTPlanDialectAttrInterfacesIncGen)

  add_mlir_tensorrt_library(${target}
    PARTIAL_SOURCES_INTENDED
    ${ARG_UNPARSED_ARGUMENTS}
    DEPENDS ${tablegen_depends})
endfunction()

# ------------------------------------------------------------------------------
# Sets `${out_var}` to the hash of the current commit. If Git is not available
# or the hash lookup fails then it sets an empty string.
# ------------------------------------------------------------------------------
function(mlir_tensorrt_find_git_hash out_var)
  if(NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set("${out_var}" "unknown" PARENT_SCOPE)
    return()
  endif()

  find_package(Git)
  if((NOT Git_FOUND) AND (NOT "${${out_var}}"))
    message(WARNING
      "Git was not found and ${out_var} was not pre-populated")
    set("${out_var}" "" PARENT_SCOPE)
    return()
  endif()

  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    RESULT_VARIABLE result
    OUTPUT_VARIABLE git_hash
    OUTPUT_STRIP_TRAILING_WHITESPACE
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  if(result)
    message(WARNING "'git rev-parse HEAD' failed: ${result}")
    set("${out_var}" "" PARENT_SCOPE)
    return()
  endif()
  set("${out_var}" "${git_hash}" PARENT_SCOPE)
endfunction()
