
# Force-set internal cache variable "MLIR_TRT_APPLY_EXTRA_CHECKS" to
# the result of the expression
# (MLIR_TRT_ENABLE_EXTRA_CHECKS AND (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")).
# This is done to avoid repeated checks when we populate target compilation options.
set(MLIR_TRT_APPLY_EXTRA_CHECKS FALSE CACHE INTERNAL "")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND MLIR_TRT_ENABLE_EXTRA_CHECKS)
  set(MLIR_TRT_APPLY_EXTRA_CHECKS TRUE CACHE INTERNAL "")
endif()

if(MLIR_TRT_ENABLE_WERROR)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -Wno-error")
endif()

# ------------------------------------------------------------------------------
# Applies additional compilation options to a C/CXX library or executable
# target. This only has an effect if MLIR_TRT_ENABLE_EXTRA_CHECKS is ON.
# ------------------------------------------------------------------------------

macro(mtrt_apply_extra_checks_to_target_helper target)
  target_compile_options(${target} PRIVATE
    -Wmissing-declarations
    -Wmissing-prototypes
    -Wunused
    -fstrict-flex-arrays=3
    )
endmacro()

macro(mtrt_apply_extra_check_options target)
  if(MLIR_TRT_APPLY_EXTRA_CHECKS)
    if(TARGET "obj.${target}")
      mtrt_apply_extra_checks_to_target_helper("obj.${target}")
    else()
      mtrt_apply_extra_checks_to_target_helper("${target}")
    endif()
  endif()
endmacro()

# ------------------------------------------------------------------------------
# Updates global C/CXX library or executable compilation flags.
# This only has an effect if MLIR_TRT_ENABLE_EXTRA_CHECKS is ON.
#
# Some flags enable additional warnings, others may enable additional runtime
# safety or preconditions checks.
#
# For a description of each flag, see the below reference:
# https://best.openssf.org/Compiler-Hardening-Guides/Compiler-Options-Hardening-Guide-for-C-and-C++.html
# ------------------------------------------------------------------------------
function(mtrt_update_global_c_cxx_flags)
  string(JOIN " " flags ${ARGN})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flags}" PARENT_SCOPE)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${flags}" PARENT_SCOPE)
endfunction()

if(MLIR_TRT_APPLY_EXTRA_CHECKS)
  # FORTIFY_SOURCE=3 requires O1 or higher.
  # GCC has default FORTIFY_SOURCE=2 and would require `-U_FORTIFY_SOURCE`
  # to change the value, so we just set it to 3 when using clang.
  if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug"
     AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_definitions(
      _FORTIFY_SOURCE=3
    )
  endif()

  mtrt_update_global_c_cxx_flags(
    # Make template instantiation errors more readable.
    -fdiagnostics-show-template-tree
    # Enable better bounds checking for trailing array members.
    -fstrict-flex-arrays=1
    # Enable runtime checks for variable-size stack allocation validity.
    -fstack-clash-protection
    # Enable runtime checks for stack-based buffer overflows.
    -fstack-protector-strong
    # Treat obsolete C constructs as errors.
    -Werror=implicit
    -Werror=incompatible-pointer-types
    -Werror=int-conversion
  )

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    mtrt_update_global_c_cxx_flags(
      -fcf-protection=full
      -Wself-assign
    )
  endif()
endif()

# This ensures that `__FILE__` in log printing expands to a relative path and file paths
# in debug info will expand relative to the source or binary directories.
#
#
# It adds the following global compilation flag:
#   '-fno-canonical-prefixes'
#   '-ffile-prefix-map=[project_src_root]=.'
#   '-fdebug-prefix-map=[project_src_root]=.'
# In addition, this flag is added if the build directory is not a subdirectory of the source directory:
#   '-ffile-prefix-map=[project_binary_root]=build'.
#   '-fdebug-prefix-map=[project_binary_root]=build'.
#
# This is only applied if the project is the top-level project and MLIR_TRT_RELATIVE_DEBUG_PATHS is ON.
function(mtrt_apply_relative_path_options)
  if(NOT COMMAND mtrt_append_compiler_flag_if_supported)
    message(FATAL_ERROR "MTRTCompilationOptions must be included after MTRTCMakeExtras")
  endif()

  # In CMake, you can't know where the "cmake" command is invoked from, so we
  # just assume that we want make paths relative to the source directory.
  set(relative_src ".")

  # The following checks if the source directory is a prefix of the binary directory.
  cmake_path(IS_PREFIX CMAKE_SOURCE_DIR "${CMAKE_BINARY_DIR}" src_dir_is_prefix_of_bin_dir)

  set(debug_remaps "-fdebug-prefix-map=${CMAKE_SOURCE_DIR}/=")
  set(file_remaps "-ffile-prefix-map=${CMAKE_SOURCE_DIR}/=")

  if(NOT src_dir_is_prefix_of_bin_dir)
    set(debug_remaps "${debug_remaps} -fdebug-prefix-map=${CMAKE_BINARY_DIR}=build")
    set(file_remaps "${file_remaps} -ffile-prefix-map=${CMAKE_BINARY_DIR}=build")
  endif()

  mtrt_append_compiler_flag_if_supported(
    CHECK "-fdebug-prefix-map=foo=bar"
    APPEND "${debug_remaps}"
  )

  mtrt_append_compiler_flag_if_supported(
    CHECK "-ffile-prefix-map=foo=bar"
    APPEND "${file_remaps}"
  )
  mtrt_append_compiler_flag_if_supported(
    CHECK "-no-canonical-prefixes"
    APPEND "-no-canonical-prefixes"
  )

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" PARENT_SCOPE)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" PARENT_SCOPE)
endfunction()

if(PROJECT_IS_TOP_LEVEL AND MLIR_TRT_RELATIVE_DEBUG_PATHS)
  mtrt_apply_relative_path_options()
endif()
