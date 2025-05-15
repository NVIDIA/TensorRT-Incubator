add_mlir_executor_library(MLIRTensorRTTargetLua
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
  MLIRTensorRTExecutorRuntimeAPI
  MLIRTranslateLib
)
