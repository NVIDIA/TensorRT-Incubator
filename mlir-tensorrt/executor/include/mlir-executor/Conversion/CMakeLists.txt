include_directories(${MLIR_INCLUDE_DIRS})
set(LLVM_TARGET_DEFINITIONS Passes.td)
set(_TABLEGEN_ARGS )

mlir_tablegen(Passes.h.inc -gen-pass-decls -name ExecutorConversion ${_TABLEGEN_ARGS})
mtrt_add_public_tablegen_target(MLIRExecutorConversionPassIncGen)
