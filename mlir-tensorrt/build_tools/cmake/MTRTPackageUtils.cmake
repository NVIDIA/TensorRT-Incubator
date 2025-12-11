# Utilities for declaring and downloading dependencies.

#-------------------------------------------------------------------------------------
# Registers a package for later addition with nv_add_package
#
# Parameters:
#   NAME - Package name (required)
#   DOWNLOAD_NAME - Alternative download name (optional)
#   All other arguments are stored for later use
#
# Usage:
#   nv_register_package(NAME MyPackage GIT_REPOSITORY url GIT_TAG tag)
#-------------------------------------------------------------------------------------
function(nv_register_package)
  cmake_parse_arguments(ARG
    "" "NAME" "" ${ARGN})
  set(NV_PACKAGE_${ARG_NAME}_ARGS ${ARGV} PARENT_SCOPE)
  set_property(GLOBAL APPEND PROPERTY NV_CPM_PACKAGES ${ARG_NAME})
endfunction()

#-------------------------------------------------------------------------------------
# A utility for use within `PRE_ADD_HOOK` of `nv_register_package` to update the
# arguments passed to `CPMAddPackage` when `nv_add_package` is called.
#
# Parameters:
#   var - Variable name to set (required)
#   value - Value to set (required)
#
# Usage:
#   nv_set_variable_in_caller_scope(CMAKE_CXX_FLAGS "-Wno-unused-variable")
#
#-------------------------------------------------------------------------------------
macro(nv_update_append_pkg_args)
  list(APPEND "${arg_prefix}_UNPARSED_ARGUMENTS" ${ARGN})
endmacro()

#-------------------------------------------------------------------------------------
# Appends a string to the `OPTIONS` list of the current package.
# For use in `PRE_ADD_HOOK` of `nv_register_package` only.
#-------------------------------------------------------------------------------------
macro(nv_pkg_append_options options_string)
  if("${arg_prefix}" STREQUAL "")
    message(FATAL_ERROR "nv_pkg_append_options: should be used in PRE_ADD_HOOK of nv_register_package only")
  endif()
  list(APPEND "${arg_prefix}_OPTIONS" "${options_string}")
endmacro()

macro(nv_pkg_append_cxx_flags)
  string(JOIN " " _new_flags ${ARGN})
  set("${arg_prefix}_CXX_FLAGS" "${${arg_prefix}_CXX_FLAGS} ${_new_flags}")
endmacro()

macro(nv_pkg_filter_out_flags regex_pattern)
  mtrt_filter_out_flags("${arg_prefix}_CXX_FLAGS" "${regex_pattern}")
endmacro()

#-------------------------------------------------------------------------------------
# Adds a previously registered package using CPM with pre/post hooks
#
# Parameters:
#   name - Package name (required)
#
# Usage:
#   nv_add_package(MyPackage)
#
# Note: Package must have been previously registered with nv_register_package
# Note: This must be a macro, not a function, due to scoping rules and use of
# `cmake_language` to execute the pre/post package add hooks, and to ensure
# variables set by CPMAddPackage are available in the caller's scope. In particular,
# CPM will use `set(.... PARENT_SCOPE)`, and if `nv_add_package` is a function,
# the variables will not be set in the scope where `find_package` is called.
#-------------------------------------------------------------------------------------
macro(nv_add_package name)
  # Validate required arguments
  # Note: In macros, arguments are string substitutions, not variables.
  # Therefore we check if the substituted string is empty.
  if("${name}" STREQUAL "")
    message(FATAL_ERROR "nv_add_package: name parameter is required")
  endif()

  set(arg_prefix "nv_add_package_args_${name}")

  cmake_parse_arguments("${arg_prefix}"
   ""
   "POST_ADD_HOOK;PRE_ADD_HOOK;DOWNLOAD_NAME;NAME"
   "OPTIONS;DOWNLOAD_COMMMAND;PATCHES"
   ${NV_PACKAGE_${name}_ARGS})

  set("${arg_prefix}_CXX_FLAGS" "${CMAKE_CXX_FLAGS}")
  if(NOT "${${arg_prefix}_OPTIONS}" STREQUAL "")
    foreach(flag IN LISTS "${arg_prefix}_OPTIONS")
      if(flag MATCHES "^CMAKE_C(XX)?_FLAGS (.*)")
        message(FATAL_ERROR "Do not set CMAKE_C_FLAGS/CMAKE_CXX_FLAGS in OPTIONS, use PRE_ADD_HOOK instead")
      endif()
    endforeach()
  endif()

  if(NOT "${${arg_prefix}_PRE_ADD_HOOK}" STREQUAL "")
    cmake_language(EVAL CODE "${${arg_prefix}_PRE_ADD_HOOK}")
  endif()

  if(NOT "${${arg_prefix}_CXX_FLAGS}" STREQUAL "")
    nv_pkg_append_options("CMAKE_CXX_FLAGS ${${arg_prefix}_CXX_FLAGS}")
  endif()

  if(NOT "${${arg_prefix}_DOWNLOAD_COMMAND}" STREQUAL "")
    list(APPEND "${arg_prefix}_UNPARSED_ARGUMENTS"
      DOWNLOAD_COMMAND ${${arg_prefix}_DOWNLOAD_COMMAND})
  endif()

  if(NOT "${${arg_prefix}_PATCHES}" STREQUAL "")
    list(APPEND "${arg_prefix}_UNPARSED_ARGUMENTS"
      PATCHES ${${arg_prefix}_PATCHES})
  endif()

  if(NOT "${${arg_prefix}_OPTIONS}" STREQUAL "")
    list(APPEND "${arg_prefix}_UNPARSED_ARGUMENTS"
      OPTIONS ${${arg_prefix}_OPTIONS})
  endif()

  if(NOT "${${arg_prefix}_DOWNLOAD_NAME}" STREQUAL "")
    mlir_tensorrt_add_package(
      NAME "${${arg_prefix}_DOWNLOAD_NAME}"
      ${${arg_prefix}_UNPARSED_ARGUMENTS})
  else()
    mlir_tensorrt_add_package(
      NAME "${${arg_prefix}_NAME}"
      ${${arg_prefix}_UNPARSED_ARGUMENTS})
  endif()

  if(NOT "${${arg_prefix}_POST_ADD_HOOK}" STREQUAL "")
    cmake_language(EVAL CODE "${${arg_prefix}_POST_ADD_HOOK}")
  endif()
endmacro()

#-------------------------------------------------------------------------------------
# Wrapper around CPMAddPackage This functions exactly like CPMAddPackage.
#-------------------------------------------------------------------------------------
macro(mlir_tensorrt_add_package)
  cmake_parse_arguments(ARG
    "" "GIT_REPOSITORY;GIT_TAG" "" ${ARGN})

  if(ARG_GIT_TAG)
    set(_mtrt_tmp_CACHE_KEY "${ARG_GIT_TAG}")
  endif()

  if(ARG_GIT_REPOSITORY)
    list(APPEND ARG_UNPARSED_ARGUMENTS
      GIT_REPOSITORY "${ARG_GIT_REPOSITORY}"
      )
  endif()
  if(ARG_GIT_TAG)
    list(APPEND ARG_UNPARSED_ARGUMENTS
      GIT_TAG "${ARG_GIT_TAG}"
      )
  endif()

  if(_mtrt_tmp_CACHE_KEY)
    list(APPEND ARG_UNPARSED_ARGUMENTS
         CUSTOM_CACHE_KEY "${_mtrt_tmp_CACHE_KEY}")
    set(_mtrt_tmp_CACHE_KEY)
  endif()

  CPMAddPackage(
    ${ARG_UNPARSED_ARGUMENTS}
  )
endmacro()
