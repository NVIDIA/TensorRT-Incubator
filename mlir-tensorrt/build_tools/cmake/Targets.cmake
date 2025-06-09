include(build_tools/cmake/CompilationOptions.cmake)
include(CMakeParseArguments)

# --------------------------------------------------------------
# Wrapper around `add_mlir_library` for creating a MLIR library.
# We append correct compilation opts/defs and add some additional
# warnings/diagnostic configs.
# --------------------------------------------------------------
function(add_mlir_tensorrt_library target)
  set(prefix ARG)
  set(noValues "")
  set(singleValues "")
  set(multiValues "")
  cmake_parse_arguments(${prefix} "${noValues}" "${singleValues}"
                        "${multiValues}" ${ARGN})

  add_mlir_library(${target}
    ${ARG_UNPARSED_ARGUMENTS}
  )
  set_property(GLOBAL APPEND PROPERTY MLIR_TENSORRT_LIBS ${target})
  if(TARGET obj.${target})
    _mtrt_set_target_compile_defs(obj.${target})
  endif()
endfunction()

# --------------------------------------------------------------
# Wrapper around `add_mlir_tensorrt_public_c_api_library` for creating a CAPI library
# We append correct compilation opts/defs and add some additional
# warnings/diagnostic configs.
# --------------------------------------------------------------
function(add_mlir_tensorrt_public_c_api_library target)
  set(prefix ARG)
  set(noValues "")
  set(singleValues "")
  set(multiValues "")
  cmake_parse_arguments(${prefix} "${noValues}" "${singleValues}"
                        "${multiValues}" ${ARGN})
  add_mlir_public_c_api_library(${target}
    ${ARG_UNPARSED_ARGUMENTS}
  )
  set_property(GLOBAL APPEND PROPERTY MLIR_TENSORRT_LIBS ${target})
  if(TARGET obj.${target})
    _mtrt_set_target_compile_defs(obj.${target})
  endif()
endfunction()

# --------------------------------------------------------------
# Adds an upstream MLIR library target to the
# MLIR_TENSORRT_LIBS global property list to capture it as an
# implicit dependency for all final tools and compiler
# end-user products.
# --------------------------------------------------------------
function(add_mlir_tensorrt_compiler_dependency target)
  set_property(GLOBAL APPEND PROPERTY MLIR_TENSORRT_LIBS ${target})
endfunction()

# ------------------------------------------------------------------------------
# A wrapper around `add_mlir_dialect_library` that also appends the dialect
# library to the global `MLIR_TENSORRT_DIALECT_LIBS` list property.
# ------------------------------------------------------------------------------
function(add_mlir_tensorrt_dialect_library target)
  set_property(GLOBAL APPEND PROPERTY MLIR_TENSORRT_DIALECT_LIBS ${target})
  add_mlir_dialect_library(${target} ${ARGN})
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

  set(LLVM_TARGET_DEFINITIONS "${SRC_TD}")

  string(REPLACE ".td" "Attrs.h.inc" h_inc_file ${BIN_TD})
  string(REPLACE ".td" "Attrs.cpp.inc" cpp_inc_file ${BIN_TD})
  mlir_tablegen("${h_inc_file}" -gen-attrdef-decls)
  mlir_tablegen("${cpp_inc_file}" -gen-attrdef-defs)

  add_public_tablegen_target(${target}IncGen)

  add_mlir_tensorrt_library(${target}
    PARTIAL_SOURCES_INTENDED
    ${ARG_UNPARSED_ARGUMENTS}
    DEPENDS ${target}IncGen)
endfunction()