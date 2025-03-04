# LLVM `mlir_tablegen` function depends grabbing include directories
# from the scope, so we must use `include_directories` here.
add_mlir_dialect(PlanOps plan)
add_mlir_doc(PlanDialect PlanDialect Plan/ -gen-dialect-doc)
add_mlir_doc(PlanDialect PlanOps Plan/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS PlanEnums.td)
mlir_tablegen(PlanEnums.h.inc -gen-enum-decls)
mlir_tablegen(PlanEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRTensorRTPlanDialectEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS PlanAttributes.td)
mlir_tablegen(PlanAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(PlanAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRTensorRTPlanDialectAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS PlanInterfaces.td)
mlir_tablegen(PlanAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(PlanAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRTensorRTPlanDialectAttrInterfacesIncGen)
