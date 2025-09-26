add_mlir_executor_library(MLIRTensorRTExecutorUtils
  Utils.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTExecutorDialect
  MLIR_LIBS PUBLIC
    MLIRIR
)
