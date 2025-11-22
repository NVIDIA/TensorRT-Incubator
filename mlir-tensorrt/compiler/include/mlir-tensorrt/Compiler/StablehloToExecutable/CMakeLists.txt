set(_TABLEGEN_ARGS )
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name StablehloToExecutable ${_TABLEGEN_ARGS})
mtrt_add_public_tablegen_target(MLIRTensorRTStablehloToExecutableIncGen)
