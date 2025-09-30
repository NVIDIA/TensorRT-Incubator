#-------------------------------------------------------------------------------------
# Common CMake utility functions for MLIR-TensorRT projects
#
# This file consolidates common patterns and utilities used across different
# sub-projects to reduce code duplication and ensure consistency.
#-------------------------------------------------------------------------------------

include(CMakeParseArguments)

# ------------------------------------------------------------------------------
# A wrapper around 'add_public_tablegen_target' that adds a dependency on
# 'mtrt-headers'.
# ------------------------------------------------------------------------------
function(mtrt_add_public_tablegen_target target)
  add_public_tablegen_target(${target})
  add_dependencies(mtrt-headers ${target})
endfunction()

#-------------------------------------------------------------------------------------
# Adds a target to the installation set with customizable component names
#
# Parameters:
#   target - Target name (required)
#   RUNTIME_COMPONENT - Runtime component name (optional, defaults to MTRT_Runtime)
#   DEVELOPMENT_COMPONENT - Development component name (optional, defaults to MTRT_Development)
#
# Usage:
#   mtrt_add_install(MyTarget)
#   mtrt_add_install(MyTarget RUNTIME_COMPONENT MTRT_Special_Runtime)
#-------------------------------------------------------------------------------------
function(mtrt_add_install target)
  # Validate required arguments
  if(NOT target)
    message(FATAL_ERROR "mtrt_add_install: target parameter is required")
  endif()

  cmake_parse_arguments(ARG "" "RUNTIME_COMPONENT;DEVELOPMENT_COMPONENT" "" ${ARGN})

  # Set default component names if not provided
  if(NOT ARG_RUNTIME_COMPONENT)
    set(ARG_RUNTIME_COMPONENT "MTRT_Runtime")
  endif()
  if(NOT ARG_DEVELOPMENT_COMPONENT)
    set(ARG_DEVELOPMENT_COMPONENT "MTRT_Development")
  endif()

  install(TARGETS ${target}
    LIBRARY
      DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      COMPONENT ${ARG_RUNTIME_COMPONENT}
    ARCHIVE
      DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      COMPONENT ${ARG_DEVELOPMENT_COMPONENT}
    RUNTIME
      DESTINATION "${CMAKE_INSTALL_BINDIR}"
      COMPONENT ${ARG_RUNTIME_COMPONENT}
    OBJECTS
      DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      COMPONENT ${ARG_DEVELOPMENT_COMPONENT}
  )
endfunction()

#-------------------------------------------------------------------------------------#
# This function should be used instead of target_link_libraries() when linking
# MLIR libraries that are part of the MLIR dylib. For libraries that are not
# part of the dylib (like test libraries), target_link_libraries() should be
# used.
#
# When MLIR_LINK_MLIR_DYLIB is enabled, this will link against the MLIR dylib
# instead of the static libraries.
#
# Normally this doesn't need to be called direclty, it is called when
# mtrt_add_project_library is called.
#-------------------------------------------------------------------------------------
function(mtrt_target_link_mlir_libraries target type)
  if (TARGET obj.${target})
    target_link_libraries(obj.${target} ${type} ${ARGN})
    add_dependencies(obj.${target} ${ARGN})
  endif()
  if (MLIR_LINK_MLIR_DYLIB)
    target_link_libraries(${target} ${type} MLIR)
  else()
    target_link_libraries(${target} ${type} ${ARGN})
  endif()
endfunction()

#-------------------------------------------------------------------------------------
# Creates a library with consistent setup across all MLIR-TensorRT projects
#
# Parameters:
#   name - Target name (required)
#   PROJECT_NAME - Project name for global property (required)
#   LIBRARY_TYPE - Type of library (optional, defaults to "")
#   DISABLE_INSTALL - Skip installation (optional)
#   COMPONENT_PREFIX - Component prefix for installation (optional)
#   All other arguments are passed to add_mlir_library
#
# Usage:
#   mtrt_add_project_library(MyLib
#     PROJECT_NAME TensorRT
#     LIBRARY_TYPE dialect
#     SOURCES file1.cpp file2.cpp)
#-------------------------------------------------------------------------------------
function(mtrt_add_project_library name)
  # Validate required arguments
  if(NOT name)
    message(FATAL_ERROR "mtrt_add_project_library: name parameter is required")
  endif()

  cmake_parse_arguments(ARG
    "DISABLE_INSTALL"
    "PROJECT_NAME;LIBRARY_TYPE;COMPONENT_PREFIX"
    "LINK_LIBS;MLIR_LIBS;DEPENDS"
    ${ARGN})

  if(NOT ARG_PROJECT_NAME)
    message(FATAL_ERROR "mtrt_add_project_library: PROJECT_NAME parameter is required")
  endif()

  if(ARG_DEPENDS)
    list(APPEND ARG_UNPARSED_ARGUMENTS DEPENDS ${ARG_DEPENDS})
  endif()

  # Append to appropriate global property based on library type
  if(NOT ARG_LIBRARY_TYPE)
    set(ARG_LIBRARY_TYPE "LIBS")
  endif()

  # For CAPI libraries, we link MLIR statically. So we can combine the MLIR_LIBS and LINK_LIBS
  # arguments into one.
  if(ARG_MLIR_LIBS AND ARG_LIBRARY_TYPE STREQUAL "CAPI")
    list(POP_FRONT ARG_MLIR_LIBS VISIBILITY)
    list(APPEND ARG_LINK_LIBS ${ARG_MLIR_LIBS})
    set(ARG_MLIR_LIBS "")
  endif()

  # Append LINK_LIBS to the unparsed arguments
  if(ARG_LINK_LIBS)
    list(APPEND ARG_UNPARSED_ARGUMENTS LINK_LIBS ${ARG_LINK_LIBS})
  endif()

  set_property(GLOBAL APPEND PROPERTY MLIR_${ARG_PROJECT_NAME}_${ARG_LIBRARY_TYPE} ${name})
  if(ARG_LIBRARY_TYPE STREQUAL "dialect" OR ARG_LIBRARY_TYPE STREQUAL "DIALECT")
    add_mlir_dialect_library(${name} OBJECT DISABLE_INSTALL EXCLUDE_FROM_LIBMLIR ${ARG_UNPARSED_ARGUMENTS})
  else()
    add_mlir_library(${name} OBJECT DISABLE_INSTALL EXCLUDE_FROM_LIBMLIR ${ARG_UNPARSED_ARGUMENTS})
  endif()

  if(ARG_MLIR_LIBS)
    list(POP_FRONT ARG_MLIR_LIBS VISIBILITY)
    mtrt_target_link_mlir_libraries(${name} ${VISIBILITY} ${ARG_MLIR_LIBS})
  endif()

  # Add to installation unless disabled
  if(NOT ARG_DISABLE_INSTALL)
    if(ARG_COMPONENT_PREFIX)
      mtrt_add_install(${name}
        RUNTIME_COMPONENT ${ARG_COMPONENT_PREFIX}_Runtime
        DEVELOPMENT_COMPONENT ${ARG_COMPONENT_PREFIX}_Development
      )
    else()
      mtrt_add_install(${name}
        RUNTIME_COMPONENT MTRT_${ARG_PROJECT_NAME}_Runtime
        DEVELOPMENT_COMPONENT MTRT_${ARG_PROJECT_NAME}_Development
      )
    endif()
  endif()
endfunction()

#-------------------------------------------------------------------------------------
# Creates a C API library. Wraps `mtrt_add_project_library` with the appropriate
# arguments.
#
# Usage:
#   mtrt_add_capi_library(MyCAPI
#     MyAPI.cpp
#     ...
#   )
#-------------------------------------------------------------------------------------
function(mtrt_add_capi_library name)
  mtrt_add_project_library(${name}
    PROJECT_NAME MLIRTensorRT
    LIBRARY_TYPE CAPI
    OBJECT
    # TODO: Once we have a MTRT dylib target, exclude all C API libraries
    # from being linked into it.
    # EXCLUDE_FROM_LIBMTRT
    ENABLE_AGGREGATION
    ${ARGN}
  )
  set_target_properties(${name} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
  )
  target_compile_definitions(obj.${name} PRIVATE
    -DMLIR_CAPI_BUILDING_LIBRARY=1
  )
endfunction()

#-------------------------------------------------------------------------------------
# Retrieve the global property list containing list of targets associated with a
# particular project.
#
# Parameters:
#   PROJECT_NAME - Project name (required)
#   out_var - Variable name to store the result (required)
#
# Usage:
#   get_mlir_${PROJECT_NAME}_libs(MY_LIBS)
#-------------------------------------------------------------------------------------
function(mtrt_get_project_targets project_name)
  if(NOT project_name)
    message(FATAL_ERROR "mtrt_get_project_targets: project_name positional argument is required")
  endif()
  cmake_parse_arguments(ARG "" "OUT_VAR;LIBRARY_TYPE" "" ${ARGN})
  if(NOT ARG_OUT_VAR)
    message(FATAL_ERROR "mtrt_get_project_targets: OUT_VAR keyword parameter is required")
  endif()
  if(NOT ARG_LIBRARY_TYPE)
    set(ARG_LIBRARY_TYPE "LIBS")
  endif()
  get_property(targets GLOBAL PROPERTY MLIR_${project_name}_${ARG_LIBRARY_TYPE})
  set("${ARG_OUT_VAR}" ${targets} PARENT_SCOPE)
endfunction()

#-------------------------------------------------------------------------------------
# Adds a library to the global property list containing list of targets associated
# with a particular project.
#
# Arguments:
#   project_name - Project name (required)
# Keyword Parameters:
#   TARGETS - List of targets to add (required)
#   LIBRARY_TYPE - Type of library (optional, defaults to "LIBS")
#
#-------------------------------------------------------------------------------------
function(mtrt_add_project_targets project_name)
  if(NOT project_name)
    message(FATAL_ERROR "mtrt_add_project_target: project_name parameter is required")
  endif()
  cmake_parse_arguments(ARG "" "LIBRARY_TYPE" "TARGETS" ${ARGN})
  if(NOT ARG_LIBRARY_TYPE)
    set(ARG_LIBRARY_TYPE "LIBS")
  endif()
  set_property(GLOBAL APPEND PROPERTY MLIR_${project_name}_${ARG_LIBRARY_TYPE} ${ARG_TARGETS})
endfunction()

#-------------------------------------------------------------------------------------
# Configures a LIT test site configuration file for a particular project.
#
# Arguments:
#   lit_config_template - Template file to use for the LIT test site configuration
#                         file (required)
# Keyword Parameters:
#
#-------------------------------------------------------------------------------------
function(mtrt_configure_lit_test_site_config template_file)
  if(NOT template_file)
    message(FATAL_ERROR "mtrt_configure_lit_test_site_config: template_file argument is required")
  endif()

  set(lit_config_lines "# Begin generated Common LIT config")
  foreach(FEATURE IN LISTS MLIR_TRT_FEATURE_FLAGS)
    set(feature_value "${${FEATURE}}")
    string(REGEX REPLACE "(MLIR_TRT_)" ""  FEATURE "${FEATURE}")
    string(TOLOWER "${FEATURE}" FEATURE)
    if(feature_value)
      list(APPEND lit_config_lines "config.${FEATURE} = True")
      list(APPEND lit_config_lines "config.available_features.add('${FEATURE}')")
    else()
      list(APPEND lit_config_lines "config.${FEATURE} = False")
    endif()
  endforeach()
  set(MLIR_TRT_LIT_COMMON "${MLIR_TENSORRT_ROOT_DIR}/common/test/lit_cfg_common.py")
  list(APPEND lit_config_lines "
config.mlir_tensorrt_compile_time_version = '${MLIR_TRT_TENSORRT_VERSION}'

# Execute the config script to initialize the object.
def load_common_config():
  from pathlib import Path
  import sys
  common_config_path = '${MLIR_TRT_LIT_COMMON}'
  cfg_globals = dict(globals())
  cfg_globals['config'] = config
  cfg_globals['lit_config'] = lit_config
  cfg_globals['__file__'] = common_config_path
  try:
      data = Path(common_config_path).read_text()
      exec(compile(data, common_config_path, 'exec'), cfg_globals, None)
  except SystemExit:
      e = sys.exc_info()[1]
      if e.args:
          raise
  except:
      import traceback
      lit_config.fatal(
          'unable to parse config file %r, traceback: %s'
          % (common_config_path, traceback.format_exc())
      )
load_common_config()
")
  list(APPEND lit_config_lines "# End generated common LIT config")
  list(JOIN lit_config_lines "\n" MLIR_TRT_COMMON_LIT_CONFIG)
  string(REGEX REPLACE "\\.in" "" site_file ${template_file})

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/${template_file}
    ${CMAKE_CURRENT_BINARY_DIR}/${site_file}
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )
endfunction()

#-------------------------------------------------------------------------------------
# Add header installation components for a particular project.
#
# Arguments:
#   component - installation component name (required)
#
# Keyword Parameters:
#   DIRECTORIES - List of directories to install (required)
#
#-------------------------------------------------------------------------------------
function(mtrt_add_header_installation_components component)
  if(NOT component)
    message(FATAL_ERROR "mtrt_add_header_installation_components: component argument is required")
  endif()
  cmake_parse_arguments(ARG "" "" "DIRECTORIES" ${ARGN})
  if(NOT ARG_DIRECTORIES)
    message(FATAL_ERROR "mtrt_add_header_installation_components: DIRECTORIES argument is required")
  endif()
  install(DIRECTORY ${ARG_DIRECTORIES}
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT ${component}
    FILES_MATCHING
      PATTERN "*.def"
      PATTERN "*.h"
      PATTERN "*.gen"
      PATTERN "*.inc"
      PATTERN "*.td"
      PATTERN "LICENSE.TXT"
      PATTERN "CMakeFiles" EXCLUDE
      PATTERN "config.h" EXCLUDE
    )
endfunction()
