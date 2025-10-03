add_mlir_executor_library(MLIRExecutorRunnerLib
  ExecutorRunnerMain.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTExecutorRuntimeAPI
    MLIRTensorRTExecutionEngineLuaRuntime
    
  MLIR_LIBS PUBLIC
    MLIRSupport
    MLIRIR
  )
