add_mlir_executor_library(MLIRTensorRTTargetLua
  TranslateToLua.cpp
  TranslateToRuntimeExecutable.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTExecutorDialect
  MLIRTensorRTExecutorRuntimeAPI
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
)
