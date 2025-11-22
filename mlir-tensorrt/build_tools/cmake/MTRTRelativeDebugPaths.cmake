# This ensures that `__FILE__` in log printing expands to a relative path.
#
# Use:
#
# ```cmake
# include(MTRTRelativeDebugPaths)
# ```
#
# It adds compilation options (if supported) to the project:
# '-ffile-prefix-map=[project_src_root]=<srcdir>' and
# '-ffile-prefix-map=[project_binary_root]=<bin dir relative to srcdir>'.
#
if(NOT COMMAND mtrt_append_compiler_flag_if_supported)
  message(FATAL_ERROR "MTRTRelativeDebugPaths must be included after MTRTCMakeExtras")
endif()

# In CMake, you can't know where the "cmake" command is invoked from, so we
# just assume that we want make paths relative to the source directory.
set(relative_src ".")
cmake_path(RELATIVE_PATH CMAKE_BINARY_DIR BASE_DIRECTORY "${CMAKE_SOURCE_DIR}" OUTPUT_VARIABLE relative_bin)

mtrt_append_compiler_flag_if_supported(
  CHECK "-fdebug-prefix-map=foo=bar"
  APPEND "-fdebug-prefix-map=${CMAKE_BINARY_DIR}=${relative_bin} -fdebug-prefix-map=${CMAKE_SOURCE_DIR}=${relative_src}"
)
mtrt_append_compiler_flag_if_supported(
  CHECK "-ffile-prefix-map=foo=bar"
  APPEND "-ffile-prefix-map=${CMAKE_BINARY_DIR}=${relative_bin} -ffile-prefix-map=${CMAKE_SOURCE_DIR}=${relative_src}"
)
mtrt_append_compiler_flag_if_supported(
  CHECK "-no-canonical-prefixes"
  APPEND "-no-canonical-prefixes"
)
