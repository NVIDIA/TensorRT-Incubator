include_directories(${MLIR_INCLUDE_DIRS})
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Plan)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Plan)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Plan)
mtrt_add_public_tablegen_target(MLIRTensorRTPlanTransformsPassIncGen)

add_mlir_tensorrt_doc(PlanPassesDoc
  SRC Passes.td
  OUTPUT_FILE docs/Passes/PlanPasses.md
  COMMAND -gen-pass-doc
  )
