include_directories(${MLIR_INCLUDE_DIRS})
set(_TABLEGEN_ARGS -gen-pass-decls -name TensorRTTranslation)
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc ${_TABLEGEN_ARGS})
mtrt_add_public_tablegen_target(MLIRTensorRTTranslationPassIncGen)

add_subdirectory(TensorRTEncodingOpInterface)
