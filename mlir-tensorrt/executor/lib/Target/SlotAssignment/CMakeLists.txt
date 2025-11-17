add_mlir_executor_library(MLIRTensorRTSlotAssignment
  SlotAssignment.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTExecutorDialect
  MLIR_LIBS PUBLIC
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRSupport
)
