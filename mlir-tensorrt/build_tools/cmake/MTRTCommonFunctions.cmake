#-------------------------------------------------------------------------------------
# Common CMake utility functions for MLIR-TensorRT projects
#
# This file consolidates common patterns and utilities used across different
# sub-projects to reduce code duplication and ensure consistency.
#-------------------------------------------------------------------------------------

include(CMakeParseArguments)
include(MTRTInterfaceClosure)

# ------------------------------------------------------------------------------
# A wrapper around 'add_public_tablegen_target' that adds a dependency on
# 'mtrt-headers'.
# ------------------------------------------------------------------------------
function(mtrt_add_public_tablegen_target target)
  add_public_tablegen_target(${target})
  add_dependencies(mtrt-headers ${target})
endfunction()

#-------------------------------------------------------------------------------------
# Add installation targets. These targets are used to install a particular CMake
# component. We also add the correct dependencies so that `ninja -C build
# install-[target name]` will force building the `[target name]` target and all
# its dependencies followed by its installation. We can then compose these
# targets to create full custom installation targets comprised of a fixed number
# of components into a specific installation prefix.
#
# Positional Arguments:
#   target - Target name (required, should be prefixed with 'install-')
#
# Keyword Parameters:
#   COMPONENT - Installation component name (required)
#   PREFIX    - Installation prefix (optional, defaults to ${CMAKE_INSTALL_PREFIX})
#   BINARY_DIR - Binary directory (optional, defaults to ${PROJECT_BINARY_DIR})
#   DEPENDS   - List of dependencies (optional)
#-------------------------------------------------------------------------------------
function(mtrt_add_install_target target)
  cmake_parse_arguments(ARG "" "COMPONENT;PREFIX;BINARY_DIR" "DEPENDS" ${ARGN})
  if(NOT target MATCHES "^install-")
    message(FATAL_ERROR "mtrt_add_install_target: target argument should be prefixed with 'install-'")
  endif()
  if(NOT ARG_COMPONENT)
    message(FATAL_ERROR "mtrt_add_install_target: COMPONENT argument is required")
  endif()
  if(NOT ARG_PREFIX)
    set(ARG_PREFIX "${CMAKE_INSTALL_PREFIX}")
  endif()
  if(NOT ARG_BINARY_DIR)
    set(ARG_BINARY_DIR "${PROJECT_BINARY_DIR}")
  endif()
  set(cmd_args "--install" "${ARG_BINARY_DIR}" "--component" "${ARG_COMPONENT}"
               "--prefix" "${ARG_PREFIX}")
  # Separate dependencies into file and target dependencies.
  # "add_custom_target" can only handle file dependencies; target dependencies
  # are handled by "add_dependencies".
  if(NOT ARG_DEPENDS AND TARGET ${ARG_COMPONENT})
    list(APPEND ARG_DEPENDS ${ARG_COMPONENT})
  endif()
  set(file_dependencies)
  set(target_dependencies)
  foreach(dependency ${ARG_DEPENDS})
    if(TARGET ${dependency})
      list(APPEND target_dependencies ${dependency})
    else()
      list(APPEND file_dependencies ${dependency})
    endif()
  endforeach()
  # Add install targets with and without stripping.
  add_custom_target(${target}
    DEPENDS ${file_dependencies}
    COMMAND "${CMAKE_COMMAND}" ${cmd_args}
    USES_TERMINAL
  )
  add_custom_target(${target}-stripped
    DEPENDS ${file_dependencies}
    COMMAND "${CMAKE_COMMAND}" ${cmd_args} --strip
    USES_TERMINAL
  )
  if(target_dependencies)
    add_dependencies(${target} ${target_dependencies})
    add_dependencies(${target}-stripped ${target_dependencies})
  endif()
endfunction()

#-------------------------------------------------------------------------------------
# Adds a target to the installation.
# If the installation component name is not specified, it is the same as the
# target name.
#
# If the EXPORT argument is specified, the target will be exported to the
# MTRTTargets export set.
#
# Parameters:
#   target - Target name (required)
#   EXPORT - Export set name (optional, defaults to "MTRTTargets")
#   COMPONENT - Installation component name (optional, defaults to the target name)
# Usage:
#   mtrt_add_install(MyTarget)
#-------------------------------------------------------------------------------------
function(mtrt_add_install target)
  cmake_parse_arguments(ARG "" "EXPORT;COMPONENT" "" ${ARGN})
  # Validate required arguments
  if(NOT target)
    message(FATAL_ERROR "mtrt_add_install: target parameter is required")
  endif()

  if(NOT ARG_EXPORT)
    set(ARG_EXPORT "MTRTTargets")
  endif()
  if(NOT ARG_COMPONENT)
    set(ARG_COMPONENT ${target})
  endif()

  install(TARGETS ${target}
    COMPONENT ${ARG_COMPONENT}
    EXPORT ${ARG_EXPORT}
    LIBRARY
      DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    ARCHIVE
      DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    RUNTIME
      DESTINATION "${CMAKE_INSTALL_BINDIR}"
    OBJECTS
      DESTINATION lib${LLVM_LIBDIR_SUFFIX}
  )
  mtrt_add_install_target(install-${target}
    COMPONENT ${ARG_COMPONENT}
    PREFIX "${CMAKE_INSTALL_PREFIX}"
    DEPENDS ${target}
    BINARY_DIR "${PROJECT_BINARY_DIR}"
  )
endfunction()

#-------------------------------------------------------------------------------------
# This function should be used instead of target_link_libraries() when linking
# MLIR libraries that are part of the MLIR dylib. For libraries that are not
# part of the dylib (like test libraries), target_link_libraries() should be
# used.
#
# When MTRT_LINK_MLIR_DYLIB is enabled, this will link against the MLIR dylib
# instead of the static libraries.
#
# Normally this doesn't need to be called direclty, it is called when
# mtrt_add_project_library is called.
#-------------------------------------------------------------------------------------
function(mtrt_target_link_mlir_libraries target type)
  if (TARGET obj.${target})
    target_link_libraries(obj.${target} PRIVATE ${ARGN})
    add_dependencies(obj.${target} ${ARGN})
  endif()
  if (MLIR_TRT_LINK_MLIR_DYLIB)
    target_link_libraries(${target} ${type} MLIR)
  else()
    target_link_libraries(${target} ${type} ${ARGN})
  endif()
endfunction()

#-------------------------------------------------------------------------------------
# This function should be used instead of target_link_libraries() when linking
# MLIR-TensorRT libraries into a tool executable or into a library that is
# excluded from libMTRT.
#-------------------------------------------------------------------------------------
function(mtrt_target_link_mtrt_libraries target type)
  if (TARGET obj.${target})
    target_link_libraries(obj.${target} PRIVATE ${ARGN})
    add_dependencies(obj.${target} ${ARGN})
  endif()
  if (MLIR_TRT_LINK_MTRT_DYLIB)
    target_link_libraries(${target} ${type} MTRT)
  else()
    target_link_libraries(${target} ${type} ${ARGN})
  endif()
endfunction()


#-------------------------------------------------------------------------------------
# Verify that the target does not transitively link both the MLIR dylib and
# individual libraries bundled into the MLIR dylib or the LLVM dylib and
# individual libraries bundled into the LLVM dylib. If NO_MTRT is specified,
# this will check that MTRT is NOT linked.
#
# Note that this is just a sanity check since without evaluating generator
# expressions, we cannot determine the full set of linked libraries.
#
# Returns a fatal error if such a situation is detected.
#-------------------------------------------------------------------------------------
function(mtrt_check_incorrect_dylib_usage name)
  cmake_parse_arguments(ARG "NO_MTRT" "" "" ${ARGN})
  if(NOT TARGET MLIR AND NOT TARGET LLVM)
    # Nothing to do because MLIR and LLVM targets are not built.
    return()
  endif()

  mtrt_collect_interface_link_closure(libs_closure includes_closure ${name})
  set(links_mlir_dylib OFF)
  set(links_llvm_dylib OFF)
  if("MLIR" IN_LIST libs_closure)
    set(links_mlir_dylib ON)
  endif()
  if("LLVM" IN_LIST libs_closure)
    set(links_llvm_dylib ON)
  endif()
  if(ARG_NO_MTRT AND "MTRT" IN_LIST libs_closure)
    message(FATAL_ERROR
     "ERROR: ${name} links libMTRT, which is forbidden. This indicates a linkage configuration issue.")
  endif()

  if(NOT links_mlir_dylib AND NOT links_llvm_dylib)
    # Nothing to do because the target does not link the MLIR dylib or the
    # LLVM dylib.
    return()
  endif()

  foreach(lib ${libs_closure})
    # This sequence of matches should determine that the target is an upstream MLIR library.
    if(links_mlir_dylib AND
       (${lib} MATCHES "^(MLIR[a-zA-Z]+)") AND
       (NOT (${lib} MATCHES "^(MLIRCAPI|MLIRTensorRT|MLIRExecutor|MLIRTRT|MLIRKernel)")))
      message(FATAL_ERROR "ERROR: ${name} links MLIR and ${lib}!")
    endif()
    if(links_llvm_dylib AND
       (${lib} MATCHES "^(LLVM[a-zA-Z]+)"))
      message(FATAL_ERROR "ERROR: ${name} links LLVM and ${lib}!")
    endif()
  endforeach()
endfunction()

#-------------------------------------------------------------------------------------
# Creates a library with consistent setup across all MLIR-TensorRT projects
#
# Parameters:
#   name - Target name (required)
#   PROJECT_NAME - Project name for global property (required)
#   LIBRARY_TYPE - Type of library (optional, defaults to "")
#   DISABLE_INSTALL - Skip installation (optional)
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
    "DISABLE_INSTALL;EXCLUDE_FROM_LIBMTRT;SHARED"
    "PROJECT_NAME;LIBRARY_TYPE"
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

  # Do a sanity check on naming and categorization of CAPI libraries.
  if((ARG_LIBRARY_TYPE STREQUAL "CAPI") AND (NOT (${name} MATCHES ".*CAPI.*")))
    message(FATAL_ERROR "mtrt_add_project_library: ${name} is a CAPI library but does not have 'CAPI' in the name")
  elseif((NOT ARG_LIBRARY_TYPE STREQUAL "CAPI") AND (${name} MATCHES ".*CAPI.*"))
    message(FATAL_ERROR "mtrt_add_project_library: ${name} is not a CAPI library but has 'CAPI' in the name")
  endif()

  # Check that TEST_LIBS are excluded from libMTRT.
  if(ARG_LIBRARY_TYPE STREQUAL "TEST_LIBS" AND NOT ARG_EXCLUDE_FROM_LIBMTRT)
    message(FATAL_ERROR "mtrt_add_project_library: ${name} is a test library but is not excluded from libMTRT")
  endif()

  if(ARG_LINK_LIBS)
    # Append LINK_LIBS to the unparsed arguments so it can be processed as normal.
    # However, if an artifact is meant to be excluded from LIBMTRT since it is a test library,
    # then we should filter out all internal project libraries and link against
    # libMTRT if MLIR_TRT_LINK_MTRT_DYLIB is enabled. This is because tools will
    # link against libMTRT and the test libraries, so we don't want test libraries
    # to statically link libraries that are meant to be excluded from libMTRT.
    list(APPEND ARG_UNPARSED_ARGUMENTS LINK_LIBS ${ARG_LINK_LIBS})
  endif()

  set(lib_type_args)
  if(ARG_SHARED)
    list(APPEND lib_type_args SHARED)
  else()
    list(APPEND lib_type_args OBJECT)
  endif()

  set_property(GLOBAL APPEND PROPERTY MLIR_${ARG_PROJECT_NAME}_${ARG_LIBRARY_TYPE} ${name})
  if(ARG_LIBRARY_TYPE STREQUAL "dialect" OR ARG_LIBRARY_TYPE STREQUAL "DIALECT")
    add_mlir_dialect_library(${name} ${lib_type_args} DISABLE_INSTALL EXCLUDE_FROM_LIBMLIR ${ARG_UNPARSED_ARGUMENTS})
  else()
    add_mlir_library(${name} ${lib_type_args} DISABLE_INSTALL EXCLUDE_FROM_LIBMLIR ${ARG_UNPARSED_ARGUMENTS})
  endif()

  if(ARG_MLIR_LIBS)
    list(POP_FRONT ARG_MLIR_LIBS VISIBILITY)
    mtrt_target_link_mlir_libraries(${name} ${VISIBILITY} ${ARG_MLIR_LIBS})
  endif()

  if((NOT ARG_DISABLE_INSTALL) AND
     (NOT ARG_EXCLUDE_FROM_LIBMTRT))
    set_property(GLOBAL APPEND PROPERTY MTRT_STATIC_LIBS ${name})
  endif()

  # Add to installation unless disabled
  if(NOT ARG_DISABLE_INSTALL)
    mtrt_add_install(${name})
  endif()

  mtrt_check_incorrect_dylib_usage(${name})
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
  cmake_parse_arguments(ARG "" "PROJECT_NAME" "" ${ARGN})
  if(NOT ARG_PROJECT_NAME)
    set(ARG_PROJECT_NAME MLIRTensorRT)
  endif()
  mtrt_add_project_library(${name}
    PROJECT_NAME ${ARG_PROJECT_NAME}
    LIBRARY_TYPE CAPI
    OBJECT
    EXCLUDE_FROM_LIBMTRT
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
# A wrapper around `mtrt_add_project_library` that adds a test library
# (e.g. for an MLIR test pass) which should be excluded from build artifacts that
# are meant to be distributed.
#-------------------------------------------------------------------------------------
function(mtrt_add_test_library name)
  cmake_parse_arguments(ARG "IGNORE_LINK_MTRT" "PROJECT_NAME" "LINK_LIBS;MLIR_LIBS" ${ARGN})
  if(NOT ARG_PROJECT_NAME)
    set(ARG_PROJECT_NAME MLIRTensorRT)
  endif()
  mtrt_add_project_library(${name}
    PROJECT_NAME ${ARG_PROJECT_NAME}
    LIBRARY_TYPE TEST_LIBS
    EXCLUDE_FROM_LIBMTRT
    DISABLE_INSTALL
    PARTIAL_SOURCES_INTENDED
    ${ARG_UNPARSED_ARGUMENTS}
  )
  if(ARG_LINK_LIBS)
    list(POP_FRONT ARG_LINK_LIBS VISIBILITY)
    if(NOT ARG_IGNORE_LINK_MTRT)
      # We may be link against libMTRT instead of directly to LINK_LIBS.
      mtrt_target_link_mtrt_libraries(${name} ${VISIBILITY} ${ARG_LINK_LIBS})
      # If we do link against libMTRT, we do not need to link against MLIR
      # libraries directly.
      if(MLIR_TRT_LINK_MTRT_DYLIB)
        unset(ARG_MLIR_LIBS)
      endif()
    else()
      # We will not be linking libMTRT, so link all the required
      # libs directly.
      target_link_libraries(${name} ${VISIBILITY} ${ARG_LINK_LIBS})
    endif()
  endif()
  if(ARG_MLIR_LIBS)
    # If we still have a list of MLIR libraries, link against them directly
    # or through libMLIR using the helper function.
    list(POP_FRONT ARG_MLIR_LIBS VISIBILITY)
    mtrt_target_link_mlir_libraries(${name} ${VISIBILITY} ${ARG_MLIR_LIBS})
  endif()

  set(mtrt_check_args "${name}")
  if(ARG_IGNORE_LINK_MTRT)
    list(APPEND mtrt_check_args "NO_MTRT")
  endif()
  mtrt_check_incorrect_dylib_usage(${mtrt_check_args})
endfunction()

#-------------------------------------------------------------------------------------
# Ensures that a target has all its symbols defined if it is a shared library.
# This option is only applied when sanitizers are not enabled.
#-------------------------------------------------------------------------------------
function(mtrt_require_defined_symbols target)
  if(ENABLE_ASAN OR ENABLE_TSAN OR ENABLE_UBSAN OR ENABLE_MSAN OR
     CMAKE_CXX_FLAGS MATCHES "-fsanitize" OR CMAKE_SHARED_LINKER_FLAGS MATCHES "-fsanitize")
    return()
  endif()
  if(UNIX AND NOT APPLE)
    target_link_options(${target} PRIVATE
      $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:LINKER:-z LINKER:defs>
      $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:LINKER:-z LINKER:defs>
    )
  elseif(APPLE)
    target_link_options(${target} PRIVATE
      $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:LINKER:-undefined LINKER:error>
      $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:LINKER:-undefined LINKER:error>
    )
  elseif(WIN32 AND MSVC)
    # MSVC link.exe already errors on unresolved symbols by default
  endif()
endfunction()

#-------------------------------------------------------------------------------------
# Creates a shared or static library wrapper around a set of libraries. The libraries
# are used to find underlying OBJECT libraries to bundle. The library will be shared
# unless 'STATIC' is explicitly specified.
#-------------------------------------------------------------------------------------
function(mtrt_add_aggregate_library target)
  cmake_parse_arguments(ARG "EXCLUDE_FROM_ALL;STATIC;DISABLE_INSTALL"
    "" "" ${ARGN})
  set(bundled_libs ${ARG_UNPARSED_ARGUMENTS})
  set(obj_libs)
  set(link_deps)
  set(interface_include_dirs)
  foreach(lib ${bundled_libs})
    if(TARGET obj.${lib})
      list(APPEND obj_libs $<TARGET_OBJECTS:obj.${lib}>)
      list(APPEND link_deps "$<TARGET_GENEX_EVAL:${lib},$<TARGET_PROPERTY:${lib},LINK_LIBRARIES>>")
      list(APPEND interface_include_dirs "$<TARGET_PROPERTY:${lib},INTERFACE_INCLUDE_DIRECTORIES>")
    endif()
  endforeach()

  # We want to filter `${link_deps}` to exclude all the libraries we are
  # bundling. However, `${link_deps}` is a generator expression and doesn't
  # account for direct link dependencies populated transitively.
  #
  # To work around this, we use a niche CMake feature
  # INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE. See documentation:
  # https://cmake.org/cmake/help/latest/prop_tgt/INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE.html.
  #
  # The CMake property "INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE" allows us to
  # enforce a filter on the "direct link dependencies" of any target which
  # transitively depends on "${target}.filter". This allows us to exlude the
  # non-object counter parts (e.g. the "${lib}" for each "obj.${lib}" library in
  # the set we are bundling) from being linked into the final `${target}`
  # library that we create below. This is only necessary to prevent redundant
  # linking of `${lib}` in the final `${target}` library when `${lib}` is a
  # SHARED library. This is the case if `${lib}` was explicitly declared SHARED
  # or when the global BUILD_SHARED_LIBS is ON.
  #
  # We use this feature by first creating an INTERFACE library that is just used
  # to carry the INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE information.
  #
  # There is a very important caveat to this approach:
  # INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE cannot guaruntee that a library
  # listed will NOT be linked by the final shared library. This mechanism will
  # be defeated if we transitively depend on any STATIC/SHARED/INTERFACE library
  # that has one of `${bundled_libs}` in its INTERFACE_LINK_LIBRARIES list.
  #
  # For this reason, we must be cautions about creating any INTERFACE library
  # which is in a cyclic link relationship with the set of libraries in
  # `bundled_libs`. e.g. `[bundled_libs...]` <-> `interface_lib`. This will
  # cause some libraries in `bundled_libs` to be linked redundantly into
  # `${target}` and will cause symptoms like crashes and ASAN ODR violation
  # errors.
  add_library(
    ${target}.filter
    INTERFACE
  )
  set_target_properties(${target}.filter PROPERTIES
    INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE "${bundled_libs}"
    )

  if(ARG_EXCLUDE_FROM_ALL)
    set(exclude_from_all EXCLUDE_FROM_ALL)
  else()
    unset(exclude_from_all)
  endif()

  set(lib_type SHARED)
  if(ARG_STATIC)
    set(lib_type STATIC)
  endif()

  add_library(
    ${target}
    ${exclude_from_all}
    ${lib_type}
  )
  set_target_properties(${target} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${LLVM_LIBRARY_OUTPUT_INTDIR}"
    LINKER_LANGUAGE CXX
  )
  mtrt_require_defined_symbols(${target})
  # Link the `${target}.filter`, which will populate the exclusion filter
  # that will apply to the final resolved direct link dependencies.
  target_link_libraries(${target} PRIVATE ${target}.filter ${obj_libs} ${link_deps})
  target_include_directories(${target} PUBLIC ${interface_include_dirs})
  if(NOT ARG_DISABLE_INSTALL)
    mtrt_add_install(${target})
  endif()
  mtrt_check_incorrect_dylib_usage(${target})
endfunction()

#-------------------------------------------------------------------------------------
# Creates a MLIR-TensorRT tool with proper installation and global registration.
#
# Parameters:
#   target - Target name (required)
#   All other arguments are passed to add_llvm_executable
#
# Usage:
#   add_mlir_tensorrt_tool(MyTool SOURCES file1.cpp file2.cpp)
#-------------------------------------------------------------------------------------
function(mtrt_add_tool target)
  cmake_parse_arguments(ARG "IGNORE_LINK_MTRT" "" "LINK_LIBS;MLIR_LIBS" ${ARGN})
  add_llvm_executable(${target}
    ${ARG_UNPARSED_ARGUMENTS}
  )
  llvm_update_compile_flags(${target})
  if(ARG_LINK_LIBS)
    if(NOT ARG_IGNORE_LINK_MTRT)
      list(POP_FRONT ARG_LINK_LIBS VISIBILITY)
      mtrt_target_link_mtrt_libraries(${target} ${VISIBILITY} ${ARG_LINK_LIBS})
      if(MLIR_TRT_LINK_MTRT_DYLIB)
        unset(ARG_MLIR_LIBS)
      endif()
    else()
      list(POP_FRONT ARG_LINK_LIBS VISIBILITY)
      target_link_libraries(${target} ${VISIBILITY} ${ARG_LINK_LIBS})
    endif()
  endif()
  if(ARG_MLIR_LIBS)
    list(POP_FRONT ARG_MLIR_LIBS VISIBILITY)
    mtrt_target_link_mlir_libraries(${target} ${VISIBILITY} ${ARG_MLIR_LIBS})
  endif()

  mtrt_add_install(${target})
  set(mtrt_check_args "${target}")
  if(ARG_IGNORE_LINK_MTRT)
    list(APPEND mtrt_check_args "NO_MTRT")
  endif()
  mtrt_check_incorrect_dylib_usage(${mtrt_check_args})
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
  set(ENABLE_ASSERTIONS "False")
  if(LLVM_ENABLE_ASSERTIONS)
    set(ENABLE_ASSERTIONS "True")
  endif()
  list(APPEND lit_config_lines "
config.mlir_tensorrt_compile_time_version = '${MLIR_TRT_TENSORRT_VERSION}'
config.enable_assertions = ${ENABLE_ASSERTIONS}

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

#-------------------------------------------------------------------------------------
# Adds a pybind11 extension library.
#
# Arguments:
#   target - Target name (required)
# Keyword Parameters:
#   ROOT_DIR - Binary directory where Python packages are installed into.
#   OUTPUT_DIR - Relative path from the ROOT_DIR to where the library should be
#      created.
#   EXTENSION_NAME - Extension name (required)
#   PRIVATE_LINK_LIBS - Private link libraries (required)
#-------------------------------------------------------------------------------------
function(mtrt_add_python_extension target)
  cmake_parse_arguments(ARG "DISABLE_INSTALL"
    "ROOT_DIR;OUTPUT_DIR;EXTENSION_NAME" "PRIVATE_LINK_LIBS" ${ARGN})
  if(NOT ARG_EXTENSION_NAME)
    message(FATAL_ERROR "mtrt_add_python_extension: EXTENSION_NAME argument is required")
  endif()
  if(NOT ARG_ROOT_DIR)
    message(FATAL_ERROR "mtrt_add_python_extension: ROOT_DIR argument is required")
  endif()
  if(NOT ARG_OUTPUT_DIR)
    message(FATAL_ERROR "mtrt_add_python_extension: OUTPUT_DIR argument is required")
  endif()
  pybind11_add_module(${target}
    ${ARG_UNPARSED_ARGUMENTS}
  )
  set_target_properties(
    ${target}
    PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${ARG_ROOT_DIR}/${ARG_OUTPUT_DIR}"
      OUTPUT_NAME ${ARG_EXTENSION_NAME}
      NO_SONAME ON
    )
  target_link_libraries(
    ${target}
    PRIVATE
    ${ARG_PRIVATE_LINK_LIBS}
    )
  if(APPLE OR UNIX)
    set(_origin_prefix "\$ORIGIN")
    if(APPLE)
      set(_origin_prefix "@loader_path")
    endif()
    set_target_properties(${target} PROPERTIES
      BUILD_WITH_INSTALL_RPATH OFF
      INSTALL_RPATH "${_origin_prefix}"
    )
  endif()
  if(NOT ARG_DISABLE_INSTALL)
    install(TARGETS ${target}
      COMPONENT ${target}
      DESTINATION "python_packages/${ARG_OUTPUT_DIR}"
      )
  endif()
endfunction()
