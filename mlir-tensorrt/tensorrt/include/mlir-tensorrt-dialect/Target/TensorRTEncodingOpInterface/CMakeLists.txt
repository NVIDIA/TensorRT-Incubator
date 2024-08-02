include_directories(${MLIR_INCLUDE_DIRS})


function(add_generate_enum_converters targetName)
  list(TRANSFORM MLIR_INCLUDE_DIRS PREPEND "-I" OUTPUT_VARIABLE _mlir_includes)
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/EnumConverters.inc.cpp
    COMMAND mlir-tensorrt-tblgen --gen-tensorrt-enum-converter-defs
      ${MLIR_TENSORRT_DIALECT_SOURCE_DIR}/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTEnums.td
      -I ${MLIR_TENSORRT_DIALECT_SOURCE_DIR}/include
      ${_mlir_includes}
      -o ${CMAKE_CURRENT_BINARY_DIR}/EnumConverters.inc.cpp
    DEPENDS ${MLIR_TENSORRT_DIALECT_SOURCE_DIR}/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTEnums.td
            mlir-tensorrt-tblgen
  )
  add_custom_target(${targetName} DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/EnumConverters.inc.cpp)
endfunction()

add_generate_enum_converters(MLIRTensorRTEnumConverterGen)

add_mlir_interface(TensorRTEncodingOpInterface)
