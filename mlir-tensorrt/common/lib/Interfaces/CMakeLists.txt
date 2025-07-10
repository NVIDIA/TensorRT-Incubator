
function(mlir_tensorrt_common_op_interface interface)
  set(LLVM_TARGET_DEFINITIONS
    "${MLIR_TENSORRT_COMMON_SOURCE_DIR}/include/mlir-tensorrt-common/Interfaces/${interface}.td")
  set(OUTPUT_DIR
     "${MLIR_TENSORRT_COMMON_BINARY_DIR}/include/mlir-tensorrt-common/Interfaces")

  cmake_path(SET OUTPUT_DIR NORMALIZE "${OUTPUT_DIR}")
  cmake_path(RELATIVE_PATH OUTPUT_DIR BASE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  mlir_tablegen("${OUTPUT_DIR}/${interface}.h.inc" -gen-op-interface-decls)
  mlir_tablegen("${OUTPUT_DIR}/${interface}.cpp.inc" -gen-op-interface-defs)
  add_public_tablegen_target(MLIRTensorRTCommon${interface}IncGen)
endfunction()

function(add_mlir_tensorrt_common_interface_library target)
   add_mlir_tensorrt_library("${target}"
    PARTIAL_SOURCES_INTENDED
    ${ARGN}
   )
endfunction()

mlir_tensorrt_common_op_interface(ToLoopsOpInterface)

add_mlir_tensorrt_common_interface_library(
  MLIRTensorRTCommonToLoopsOpInterface

  ToLoopsOpInterface.cpp

  DEPENDS
  MLIRTensorRTCommonToLoopsOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRTensorRTCommonIncludes
  MLIRIR
  )