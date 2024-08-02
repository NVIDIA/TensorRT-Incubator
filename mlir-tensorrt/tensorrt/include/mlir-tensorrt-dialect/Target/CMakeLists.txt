include_directories(${MLIR_INCLUDE_DIRS})
set(_TABLEGEN_ARGS -gen-pass-decls -name TensorRTTranslation)
set(LLVM_TARGET_DEFINITIONS Passes.td)
if(MLIR_TRT_TARGET_LUA)
  list(APPEND _TABLEGEN_ARGS -DMLIR_TRT_TARGET_LUA)
endif()
mlir_tablegen(Passes.h.inc ${_TABLEGEN_ARGS})
add_public_tablegen_target(MLIRTensorRTTranslationPassIncGen)

add_subdirectory(TensorRTEncodingOpInterface)

