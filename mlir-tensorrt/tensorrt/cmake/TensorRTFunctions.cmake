# --------------------------------------------------------------
# Creates `targetName` that invokes mlir-tensorrt-tblgen
# on a [dialect]Ops.td file to generate implementations for the
# TensorRTEncodingOpInterface's encodeOp interface method.
# --------------------------------------------------------------
function(add_tensorrt_encoding_def_gen targetName inputFileName outputFileName )
    list(TRANSFORM MLIR_INCLUDE_DIRS PREPEND "-I" OUTPUT_VARIABLE _mlir_includes)
    add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${outputFileName}"
    COMMAND mlir-tensorrt-tblgen --gen-tensorrt-layer-add-defs
      "${inputFileName}"
      -I "${MLIR_TENSORRT_ROOT_DIR}/include"
      -I "${MLIR_TENSORRT_ROOT_DIR}/tensorrt/include"
      ${_mlir_includes}
      -o "${CMAKE_CURRENT_BINARY_DIR}/${outputFileName}"
    DEPENDS "${inputFileName}" mlir-tensorrt-tblgen
    )
    add_custom_target(${targetName} DEPENDS
      "${CMAKE_CURRENT_BINARY_DIR}/${outputFileName}")
  endfunction()
