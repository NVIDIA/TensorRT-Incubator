# LLVM `mlir_tablegen` function depends grabbing include directories
# from the scope, so we must use `include_directories` here.
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${MLIR_TENSORRT_ROOT_DIR}/executor/include)

add_mlir_dialect(ExecutorOps executor)
if(COMMAND add_mlir_tensorrt_doc)
  add_mlir_tensorrt_doc(ExecutorDialect
    SRC ExecutorOps.td
    OUTPUT_FILE docs/Dialects/ExecutorDialect.md
    COMMAND -gen-dialect-doc -dialect=executor
    )
endif()

# Op Interfaces
add_mlir_interface(ExecutorInterfaces)

set(LLVM_TARGET_DEFINITIONS ExecutorEnums.td)
mlir_tablegen(ExecutorEnums.h.inc -gen-enum-decls)
mlir_tablegen(ExecutorEnums.cpp.inc -gen-enum-defs)
mtrt_add_public_tablegen_target(MLIRTensorRTExecutorDialectEnumsIncGen)


set(LLVM_TARGET_DEFINITIONS ExecutorAttributes.td)
mlir_tablegen(ExecutorAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(ExecutorAttributes.cpp.inc -gen-attrdef-defs)
mtrt_add_public_tablegen_target(MLIRTensorRTExecutorDialectAttributesIncGen)
