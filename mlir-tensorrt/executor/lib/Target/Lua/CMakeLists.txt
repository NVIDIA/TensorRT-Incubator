add_mlir_executor_library(MLIRTensorRTTargetLua
  TranslateToLua.cpp
  TranslateToRuntimeExecutable.cpp

  LINK_LIBS PUBLIC
    MLIRExecutorSerializationUtils
    MLIRTensorRTExecutorDialect
    MLIRTensorRTRuntimeExecutableInterface
    MLIRTensorRTSlotAssignment
  MLIR_LIBS PUBLIC
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRSupport
    MLIRTranslateLib
)
