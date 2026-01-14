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
    MLIRTensorRTStreamSchedulableOpInterface

  MLIR_LIBS PUBLIC
    MLIRCastInterfaces
    MLIRControlFlowInterfaces
    MLIRDialect
    MLIRDialectUtils
    MLIRDLTIDialect
    MLIRFuncDialect
    MLIRInferTypeOpInterface
    MLIRIR
    MLIRMemorySlotInterfaces
    MLIRSupport
    MLIRTransformUtils
    MLIRUBDialect
)
