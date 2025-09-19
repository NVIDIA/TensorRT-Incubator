# This file contains miscellaneous CMake functions that augment the native CMake language
# and are used across the project.

#-------------------------------------------------------------------------------------
# Conditionally appends a value to multiple lists based on a condition
#
# This function appends a given value to one or more lists if the specified condition
# evaluates to true. The lists must be passed by name and will be modified in the
# caller's scope.
#
# Parameters:
#   condition - Boolean condition to evaluate (required)
#   value     - Value to append to the lists (required)
#   ARGN      - Names of lists to append to (at least one required)
#
# Usage:
#   set(MY_FLAGS)
#   set(MY_DEFINES)
#   mtrt_append_lists_if(ENABLE_FEATURE "-DFEATURE_ENABLED" MY_FLAGS MY_DEFINES)
#
# Note: The condition parameter should be a variable name or expression that evaluates
#       to a boolean value (TRUE/FALSE, ON/OFF, YES/NO, or 1/0).
#-------------------------------------------------------------------------------------
function(mtrt_append_lists_if condition value)
  # Validate required arguments
  if(ARGC LESS 3)
    message(FATAL_ERROR "mtrt_append_lists_if: Requires at least 3 arguments (condition, value, and at least one list name)")
  endif()

  # Check if condition is empty (but allow FALSE/OFF/NO/0 values)
  if(condition STREQUAL "")
    message(FATAL_ERROR "mtrt_append_lists_if: condition parameter cannot be empty")
  endif()

  # Check if value is provided (empty values are allowed)
  if(ARGV1 STREQUAL "")
    message(WARNING "mtrt_append_lists_if: appending empty value to lists")
  endif()

  # Only append if condition evaluates to true
  if(condition)
    foreach(list_name ${ARGN})
      # Verify the list name is not empty
      if(list_name STREQUAL "")
        message(WARNING "mtrt_append_lists_if: Skipping empty list name")
        continue()
      endif()

      # Append the value to the list in parent scope
      list(APPEND ${list_name} "${value}")
      set(${list_name} "${${list_name}}" PARENT_SCOPE)
    endforeach()
  endif()
endfunction()

#-------------------------------------------------------------------------------------
# Checks if a compiler flag is supported and appends it to CMAKE_C_FLAGS and
# CMAKE_CXX_FLAGS
#
# This function checks if the given compiler flag is supported by C or C++
# (as specified by the LANGS argument), and if so, appends it to the
# specified variables to the appropriate language flag variable (CMAKE_C_FLAGS or
# CMAKE_CXX_FLAGS).
#
# Parameters:
#   CHECK  - Compiler flag to check (required)
#   APPEND - Compiler flag to append if supported (optional, use CHECK flag if not provided)
#   LANGS  - Languages to check support for (optional, default is both C and C++)
#
# Usage:
#   mtrt_append_compiler_flag_if_supported(CHECK "-fvisibility=hidden")
#   mtrt_append_compiler_flag_if_supported(
#     CHECK "-ffile-prefix-map=foo=bar"
#     APPEND "-ffile-prefix-map=${CMAKE_SOURCE_DIR}=${relative_src}"
#   )
#
#-------------------------------------------------------------------------------------
function(mtrt_append_compiler_flag_if_supported)
  cmake_parse_arguments(ARG "" "CHECK;APPEND" "LANGS" ${ARGN})

  if(NOT ARG_CHECK)
    message(FATAL_ERROR "mtrt_append_compiler_flag_if_supported: CHECK parameter is required")
  endif()

  if(NOT ARG_LANGS)
    set(ARG_LANGS "C;CXX")
  endif()

  if(NOT ARG_APPEND)
    set(ARG_APPEND "${ARG_CHECK}")
  endif()

  # Check compiler support
  include(CheckCCompilerFlag)
  include(CheckCXXCompilerFlag)

  # Generate a safe variable name from the flag
  string(REGEX REPLACE "[^A-Za-z0-9_]" "_" _flag_var "${ARG_CHECK}")
  string(TOUPPER "${_flag_var}" _flag_var)

  # Check C compiler support if C language is enabled
  if(CMAKE_C_COMPILER AND ("C" IN_LIST ARG_LANGS))
    check_c_compiler_flag("${ARG_CHECK}" "C_SUPPORTS${_flag_var}")
    if(C_SUPPORTS${_flag_var})
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ARG_APPEND}" PARENT_SCOPE)
    endif()
  endif()

  # Check C++ compiler support if C++ language is enabled
  if(CMAKE_CXX_COMPILER AND ("CXX" IN_LIST ARG_LANGS))
    check_cxx_compiler_flag("${ARG_CHECK}" "CXX_SUPPORTS${_flag_var}")
    if(CXX_SUPPORTS${_flag_var})
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARG_APPEND}" PARENT_SCOPE)
    endif()
  endif()
endfunction()
