add_mlir_library(MLIRExecutorRunnerLib
  ExecutorRunnerMain.cpp

  LINK_LIBS PUBLIC
  MLIRSupport
  MLIRTensorRTExecutorRuntimeAPI
  MLIRTensorRTExecutionEngineLuaRuntime
  MLIRIR
  )
