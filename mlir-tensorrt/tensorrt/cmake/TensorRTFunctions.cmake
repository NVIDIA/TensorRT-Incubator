# --------------------------------------------------------------
# Creates `targetName` that invokes tensorrt-tblgen
# on a [dialect]Ops.td file to generate implementations for the
# TensorRTEncodingOpInterface's encodeOp interface method.
# --------------------------------------------------------------
function(add_tensorrt_encoding_def_gen targetName inputFileName outputFileName )
    list(TRANSFORM MLIR_INCLUDE_DIRS PREPEND "-I" OUTPUT_VARIABLE _mlir_includes)
    add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${outputFileName}"
    COMMAND tensorrt-tblgen --gen-tensorrt-layer-add-defs
      "${inputFileName}"
      -I "${MLIR_TENSORRT_DIALECT_SOURCE_DIR}/include"
      ${_mlir_includes}
      -o "${CMAKE_CURRENT_BINARY_DIR}/${outputFileName}"
    DEPENDS "${inputFileName}" tensorrt-tblgen
    )
    add_custom_target(${targetName} DEPENDS
      "${CMAKE_CURRENT_BINARY_DIR}/${outputFileName}")
  endfunction()

# --------------------------------------------------------------
# Add mlir-tensorrt-dialect target to install set
# --------------------------------------------------------------
function(add_mtrtd_install target)
  install(TARGETS ${target}
    LIBRARY
      DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      COMPONENT MTRT_TensorRTDialect_Runtime
    ARCHIVE
      DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      COMPONENT MTRT_TensorRTDialect_Development
    RUNTIME
      DESTINATION "${CMAKE_INSTALL_BINDIR}"
      COMPONENT MTRT_TensorRTDialect_Runtime
    OBJECTS
      DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      COMPONENT MTRT_TensorRTDialect_Development
  )
endfunction()

# --------------------------------------------------------------
# Add includes to mlir-tensorrt-dialect library
# --------------------------------------------------------------
function(populate_mtrtd_includes_helper name relationship)
  target_include_directories(${name} ${relationship}
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>)
endfunction()

function(populate_mtrtd_includes name)
  get_target_property(type ${name} TYPE)
  if (${type} STREQUAL "INTERFACE_LIBRARY")
    populate_mtrtd_includes_helper(${name}
      INTERFACE)
  else()
    populate_mtrtd_includes_helper(${name}
      PUBLIC)
  endif()
  if(TARGET obj.${name})
    populate_mtrtd_includes_helper(obj.${name}
      PUBLIC)
  endif()
endfunction()

# --------------------------------------------------------------
# Wrapper around `add_mlir_library`
# --------------------------------------------------------------
function(add_mtrtd_library name)
  set_property(GLOBAL APPEND PROPERTY MTRTD_LIBS ${name})
  add_mlir_library(${name} DISABLE_INSTALL ${ARGN})
  populate_mtrtd_includes(${name})
  add_mtrtd_install(${name})
endfunction()

