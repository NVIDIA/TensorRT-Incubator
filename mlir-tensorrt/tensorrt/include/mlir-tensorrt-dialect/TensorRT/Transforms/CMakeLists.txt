include_directories(${MLIR_INCLUDE_DIRS})
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TensorRT)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix TensorRT)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix TensorRT)
mtrt_add_public_tablegen_target(MLIRTensorRTTransformsPassIncGen)

if(COMMAND add_mlir_tensorrt_doc)
  add_mlir_tensorrt_doc(TensorRTPassesDoc
    SRC Passes.td
    OUTPUT_FILE docs/Passes/TensorRTPasses.md
    COMMAND -gen-pass-doc
    )
endif()
