set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

# Add the plugin library
function(add_pjrt_library name)
  add_library(${name} SHARED
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

  # Write the header file declaring the symbol visibility macros.
  include(GenerateExportHeader)
  generate_export_header(${name} BASE_NAME MLIR_TRT_PJRT)

  # On Linux, disable re-export of any static linked libraries that
  # came through.
  target_link_options(${name} PRIVATE
    $<$<PLATFORM_ID:Linux>:LINKER:--exclude-libs,ALL>
  )


  install(TARGETS ${name}
    LIBRARY DESTINATION
    "python_packages/mlir_tensorrt_jax/mlir_tensorrt_jax"
  )
endfunction()

add_pjrt_library(mlir-tensorrt-pjrt)
