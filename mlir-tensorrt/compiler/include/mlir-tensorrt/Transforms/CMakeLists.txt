include_directories(${MLIR_INCLUDE_DIRS})
set(LLVM_TARGET_DEFINITIONS Passes.td)
set(_TABLEGEN_ARGS -gen-pass-decls -name MLIRTensorRTGenericTransforms)

# Certain passes are predicated on the availability of third party dialects.
if(MLIR_TRT_ENABLE_HLO)
  list(APPEND _TABLEGEN_ARGS -DMLIR_TENSORRT_ENABLE_HLO)
endif()

mlir_tablegen(Passes.h.inc ${_TABLEGEN_ARGS})
mtrt_add_public_tablegen_target(MLIRTensorRTGenericTransformPassIncGen)
