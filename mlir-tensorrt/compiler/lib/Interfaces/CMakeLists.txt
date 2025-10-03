
function(add_mlir_tensorrt_interface_library target)
  cmake_parse_arguments(ARG "" "ATTR;OP;TD" "" ${ARGN})
  cmake_path(SET ARG_TD
    NORMALIZE
    "${PROJECT_SOURCE_DIR}/compiler/include/${ARG_TD}")

  set(LLVM_TARGET_DEFINITIONS "${ARG_TD}")

  if(ARG_ATTR)
    cmake_path(SET ARG_ATTR
      NORMALIZE
      "${PROJECT_BINARY_DIR}/compiler/include/${ARG_ATTR}")
    cmake_path(RELATIVE_PATH ARG_ATTR
        BASE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    mlir_tablegen("${ARG_ATTR}.h.inc" -gen-attr-interface-decls)
    mlir_tablegen("${ARG_ATTR}.cpp.inc" -gen-attr-interface-defs)
  endif()

  if(ARG_OP)
    cmake_path(SET ARG_OP
      NORMALIZE
      "${PROJECT_BINARY_DIR}/compiler/include/${ARG_OP}")
    cmake_path(RELATIVE_PATH ARG_OP
        BASE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    mlir_tablegen("${ARG_OP}.h.inc" -gen-op-interface-decls)
    mlir_tablegen("${ARG_OP}.cpp.inc" -gen-op-interface-defs)
  endif()

  mtrt_add_public_tablegen_target(${target}IncGen)

  add_mlir_tensorrt_library(${target}
    PARTIAL_SOURCES_INTENDED
    ${ARG_UNPARSED_ARGUMENTS}
    DEPENDS ${target}IncGen)
endfunction()


add_mlir_tensorrt_interface_library(MLIRTensorRTBufferizationScopeInterface
  BufferizationScopeInterface.cpp

  TD
    "mlir-tensorrt/Interfaces/BufferizationScopeInterface.td"
  OP
    "mlir-tensorrt/Interfaces/BufferizationScopeInterface"

  MLIR_LIBS PUBLIC
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
  )

add_mlir_tensorrt_interface_library(MLIRTensorRTInferTensorValueRangeInterface
  InferTensorValueRangeInterface.cpp

  TD
    "mlir-tensorrt/Interfaces/InferTensorValueRangeInterface.td"
  OP
    "mlir-tensorrt/Interfaces/InferTensorValueRangeInterface"

  MLIR_LIBS PUBLIC
  MLIRInferIntRangeInterface
  )
