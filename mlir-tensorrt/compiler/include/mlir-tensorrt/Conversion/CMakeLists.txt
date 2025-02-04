include_directories(${MLIR_INCLUDE_DIRS})
set(LLVM_TARGET_DEFINITIONS Passes.td)
set(_TABLEGEN_ARGS )

# Certain passes are predicated on the availability of third party dialects.
if(MLIR_TRT_ENABLE_HLO)
  list(APPEND _TABLEGEN_ARGS -DMLIR_TENSORRT_ENABLE_HLO)
endif()
if(MLIR_TRT_ENABLE_EXECUTOR)
  list(APPEND _TABLEGEN_ARGS -DMLIR_TENSORRT_ENABLE_EXECUTOR)
endif()

mlir_tablegen(Passes.h.inc -gen-pass-decls -name MLIRTensorRTConversion ${_TABLEGEN_ARGS})
add_public_tablegen_target(MLIRTensorRTConversionPassIncGen)

add_mlir_tensorrt_doc(ConversionPasses
  SRC Passes.td
  OUTPUT_FILE docs/Passes/ConversionPasses.md
  COMMAND -gen-pass-doc ${_TABLEGEN_ARGS}
  )

