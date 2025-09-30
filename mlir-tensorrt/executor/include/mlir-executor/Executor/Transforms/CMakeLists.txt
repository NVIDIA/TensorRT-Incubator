include_directories(${MLIR_INCLUDE_DIRS})
set(LLVM_TARGET_DEFINITIONS Passes.td)
set(_TABLEGEN_ARGS -gen-pass-decls -name ExecutorTransforms)

mlir_tablegen(Passes.h.inc ${_TABLEGEN_ARGS})
mtrt_add_public_tablegen_target(MLIRTensorRTExecutorTransformsPassIncGen)

if(COMMAND add_mlir_tensorrt_doc)
  add_mlir_tensorrt_doc(ExecutorPassesDoc
    SRC Passes.td
    OUTPUT_FILE docs/Passes/ExecutorPasses.md
    COMMAND -gen-pass-doc
    )
endif()
