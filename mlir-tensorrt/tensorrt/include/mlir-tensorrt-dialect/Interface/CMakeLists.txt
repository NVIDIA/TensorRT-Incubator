set(LLVM_TARGET_DEFINITIONS TensorKindOpInterface.td)
mlir_tablegen(TensorKindOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(TensorKindOpInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRTensorRTTensorKindOpInterfaceIncGen)


set(LLVM_TARGET_DEFINITIONS "TensorKindOpInterface.td")
mlir_tablegen("TensorKindAttrInterface.h.inc" -gen-attr-interface-decls)
mlir_tablegen("TensorKindAttrInterface.cpp.inc" -gen-attr-interface-defs)
add_public_tablegen_target("MLIRTensorRTTensorKindAttrInterfaceIncGen")