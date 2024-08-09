add_mlir_executor_library(MLIRTensorRTExecutorUtils
  Utils.cpp

  LINK_LIBS
  MLIRIR
  MLIRTensorRTExecutorDialect
)
