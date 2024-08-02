# LLVM `mlir_tablegen` function depends grabbing include directories
# from the scope, so we must use `include_directories` here.
add_mlir_dialect(TensorRTOps tensorrt)

if(COMMAND add_mlir_tensorrt_doc)
  add_mlir_tensorrt_doc(TensorRTDialect
    SRC TensorRTOps.td
    OUTPUT_FILE docs/Dialects/TensorRTDialect.md
    COMMAND -gen-dialect-doc -dialect=tensorrt
  )
endif()

set(LLVM_TARGET_DEFINITIONS TensorRTEnums.td)
mlir_tablegen(TensorRTEnums.h.inc -gen-enum-decls)
mlir_tablegen(TensorRTEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRTensorRTDialectEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS TensorRTAttributes.td)
mlir_tablegen(TensorRTAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(TensorRTAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRTensorRTDialectAttributesIncGen)

# Op Interfaces
add_mlir_interface(TensorRTInterfaces)

# Attr Interfaces
set(LLVM_TARGET_DEFINITIONS TensorRTInterfaces.td)
mlir_tablegen(TensorRTAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(TensorRTAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRTensorRTDialectAttrInterfacesIncGen)
