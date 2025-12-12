# Hide symbols by default. We only want to export the "GetPjrtApi" top level
# function.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

# Add the plugin library
function(add_pjrt_library name is_shared)
  add_library(${name} ${ARGN}
    API.cpp
    "${MLIR_TRT_XLA_SOURCE_DIR}/xla/pjrt/c/pjrt_c_api.h"
  )
  llvm_update_compile_flags(${name})
  target_compile_options(${name} PRIVATE
    -Wno-unused-variable
  )
  target_link_libraries(${name}
    PRIVATE
    MLIRTensorRTPJRTImplementation
    XLAProtobufLibs
    )
  target_link_libraries(${name}
    PUBLIC
    PJRT_CApiHeaders
    )
  target_include_directories(${name} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

  # Output the library in the PJRT build directory for
  # compatability with previous build system org.
  set_target_properties(${name} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
    )

  mtrt_require_defined_symbols(${name})

  if(is_shared)
    # Write the header file declaring the symbol visibility macros.
    include(GenerateExportHeader)
    generate_export_header(${name} BASE_NAME MLIR_TRT_PJRT)
  endif()

  if(is_shared)
    install(TARGETS ${name}
      LIBRARY DESTINATION
      "python_packages/mlir_tensorrt_jax/mlir_tensorrt_jax"
    )
  endif()
endfunction()

add_pjrt_library(mlir-tensorrt-pjrt 1 SHARED)

# Build static library just for testing use-cases.
add_pjrt_library(mlir-tensorrt-pjrt-static 0)
set_property(TARGET mlir-tensorrt-pjrt-static
  PROPERTY EXCLUDE_FROM_ALL TRUE)
