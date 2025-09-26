add_mlir_executor_library(MLIRTensorRTTargetLua
  LuaAllocation.cpp
  TranslateToLua.cpp
  TranslateToRuntimeExecutable.cpp

  LINK_LIBS PUBLIC
    MLIRExecutorSerializationUtils
    MLIRTensorRTExecutorDialect
    MLIRTensorRTRuntimeExecutableInterface
  MLIR_LIBS PUBLIC
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRSupport
    MLIRTranslateLib
)
