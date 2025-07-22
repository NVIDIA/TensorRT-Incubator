set(LLVM_TARGET_DEFINITIONS StatusEnums.td)
tablegen(MTRT StatusEnums.h.inc --gen-custom-enum-defs)
tablegen(MTRT StatusEnums.c.h.inc --gen-custom-enum-c-defs)
add_public_tablegen_target(MLIRTensorRTStatusEnumIncGen)
