#-------------------------------------------------------------------------------------
# MLIR-TensorRT Feature Flag Management
#
# This file provides utilities for declaring and managing feature flags across
# the project. Feature flags are used to conditionally enable/disable functionality
# at configure time.
#-------------------------------------------------------------------------------------

# Initialize the list of feature flags
set(MLIR_TRT_FEATURE_FLAGS "")

#-------------------------------------------------------------------------------------
# Defines a MLIR-TensorRT project option and adds it to feature flags
#
# Parameters:
#   name - Option name (required)
#   description - Option description (required)
#   default - Default value (required)
#
# Usage:
#   mtrt_option(MLIR_TRT_ENABLE_FEATURE "Enable feature" ON)
#
# Note: This is a macro to ensure options are defined in the calling scope
#-------------------------------------------------------------------------------------
macro(mtrt_option name description default)
  # Validate required arguments (macro arguments are string substitutions)
  if("${name}" STREQUAL "")
    message(FATAL_ERROR "mtrt_option: name parameter is required")
  endif()
  if("${description}" STREQUAL "")
    message(FATAL_ERROR "mtrt_option: description parameter is required")
  endif()
  if("${default}" STREQUAL "")
    message(FATAL_ERROR "mtrt_option: default parameter is required")
  endif()

  option(${name} "${description}" ${default})
  list(APPEND MLIR_TRT_FEATURE_FLAGS ${name})
endmacro()

#-------------------------------------------------------------------------------------
# Generates a header file containing convenience macros for each feature flag
#
# Usage:
#   mtrt_write_feature_flags_header()
#
# Note: Creates Features.h in the build directory with IF_FEATURE(code) macros
#-------------------------------------------------------------------------------------
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
      if(${${FEATURE}})
        file(APPEND "${feature_flags_header}" "#ifndef ${FEATURE}\n")
        file(APPEND "${feature_flags_header}" "#define ${FEATURE}\n")
        file(APPEND "${feature_flags_header}" "#endif // ${FEATURE}\n")
        file(APPEND "${feature_flags_header}" "#define IF_${FEATURE}(code) do { code } while (0)\n")
      else()
        file(APPEND "${feature_flags_header}" "#define IF_${FEATURE}(code) do {} while (0)\n")
      endif()
  endforeach()
  file(APPEND "${feature_flags_header}" "#endif // MLIR_TENSORRT_FEATURES_H\n")
endfunction()
