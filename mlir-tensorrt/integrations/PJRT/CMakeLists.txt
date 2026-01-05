include(${CMAKE_CURRENT_LIST_DIR}/PJRTConfig.cmake)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

#-------------------------------------------------------------------------------
# Project Setup
#-------------------------------------------------------------------------------

set(MLIR_TRT_PJRT_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(MLIR_TENSORRT_PJRT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

#-------------------------------------------------------------------------------
# Dependencies
#-------------------------------------------------------------------------------
function(find_pjrt_c_headers)
  # Confirm that we can find the header at configure-time.
  find_file(xla_pjrt_c_api_header pjrt_c_api.h
    HINTS ${MLIR_TRT_XLA_SOURCE_DIR}/xla/pjrt/c
    NO_PACKAGE_ROOT_PATH
    NO_CMAKE_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH
  )
  add_library(PJRT_CApiHeaders INTERFACE IMPORTED)
  target_include_directories(PJRT_CApiHeaders INTERFACE
    $<BUILD_INTERFACE:${MLIR_TRT_XLA_SOURCE_DIR}>
  )
endfunction()

find_pjrt_c_headers()

find_package(absl REQUIRED)
find_package(Protobuf REQUIRED)

#-------------------------------------------------------------------------------
# Project Targets
#-------------------------------------------------------------------------------
include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include)

# PJRT header can only compile without warnings under Clang, because
# it hits this difference between clang and gcc:
# https://stackoverflow.com/questions/15537023/declaration-of-method-changes-meaning-of-symbol
#
# To solve the issue, we need '-fpermisive' with GCC.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  add_compile_options(-fpermissive )
endif()

add_subdirectory(lib)

# This target sets up a convencience script that forwards all args to `python3`
# but also sets environment variables required to use JAX with our PJRT plugin
# integration enabled.
set(PJRT_PLUGIN_PATH "$<TARGET_FILE:mlir-tensorrt-pjrt>")
set(PY_CMD "python3 $@")
configure_file(mlir-trt-jax-py.in mlir-trt-jax-py.in @ONLY)
file(GENERATE OUTPUT ${MLIR_TENSORRT_ROOT_BINARY_DIR}/bin/mlir-trt-jax-py
  INPUT ${CMAKE_CURRENT_BINARY_DIR}/mlir-trt-jax-py.in
  FILE_PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)

set(PJRT_PLUGIN_PATH "$<TARGET_FILE:mlir-tensorrt-pjrt>")
set(PY_CMD "lldb python3 -- $@")
configure_file(mlir-trt-jax-py.in mlir-trt-jax-py-dbg.in @ONLY)
file(GENERATE OUTPUT ${MLIR_TENSORRT_ROOT_BINARY_DIR}/bin/mlir-trt-jax-py-dbg
  INPUT ${CMAKE_CURRENT_BINARY_DIR}/mlir-trt-jax-py-dbg.in
  FILE_PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)

add_subdirectory(test)

add_subdirectory(python)
