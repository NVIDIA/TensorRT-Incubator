add_mlir_executor_library(MLIRExecutorRunnerLib
  ExecutorRunnerMain.cpp

  LINK_LIBS PUBLIC
  MLIRSupport
  MLIRTensorRTExecutorRuntimeAPI
  MLIRTensorRTExecutionEngineLuaRuntime
  MLIRIR
  MLIRExecutorRuntimeCapabilities
  )
