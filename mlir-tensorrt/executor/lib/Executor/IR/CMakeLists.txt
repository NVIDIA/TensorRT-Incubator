add_mlir_dialect_library(MLIRTensorRTExecutorDialect
  Executor.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_EXECUTOR_SOURCE_DIR}/include

  DEPENDS
  MLIRExecutorOpsIncGen
  MLIRTensorRTExecutorDialectEnumsIncGen
  MLIRTensorRTExecutorDialectAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDialect
  MLIRSupport
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRControlFlowInterfaces
  MLIRMemorySlotInterfaces
  MLIRUBDialect
  MLIRDLTIDialect
  MLIRTransformUtils
  MLIRDialectUtils
)
