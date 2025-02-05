set(_TABLEGEN_ARGS )
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TensorRTToExecutable ${_TABLEGEN_ARGS})
add_public_tablegen_target(MLIRTensorRTTensorRTToExecutableIncGen)
