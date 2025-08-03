add_mlir_executor_library(MLIRTensorRTTargetLua
  LuaAllocation.cpp
  TranslateToLua.cpp
  TranslateToRuntimeExecutable.cpp

  LINK_LIBS PUBLIC
  MLIRExecutorSerializationUtils
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorRTExecutorDialect
  MLIRTensorRTRuntimeExecutableInterface
  MLIRTranslateLib
)
