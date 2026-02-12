include_directories(${MLIR_INCLUDE_DIRS})
set(LLVM_TARGET_DEFINITIONS Passes.td)
set(_TABLEGEN_ARGS )

# Certain passes are predicated on the availability of third party dialects.
if(MLIR_TRT_ENABLE_HLO)
  list(APPEND _TABLEGEN_ARGS -DMLIR_TRT_ENABLE_HLO)
endif()

mlir_tablegen(Passes.h.inc -gen-pass-decls -name CompilerPasses ${_TABLEGEN_ARGS})
mtrt_add_public_tablegen_target(MLIRTensorRTPassesIncGenPassIncGen)

add_mlir_tensorrt_doc(CompilerPasses
  SRC Passes.td
  OUTPUT_FILE docs/Passes/CompilerPasses.md
  COMMAND -gen-pass-doc ${_TABLEGEN_ARGS}
  )

