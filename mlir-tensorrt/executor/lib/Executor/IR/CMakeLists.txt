add_mlir_executor_library(MLIRTensorRTExecutorDialect
  Executor.cpp
  ExecutorAttributes.cpp

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
