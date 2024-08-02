add_mlir_library(MLIRTensorRTExecutorUtils
  Utils.cpp

  LINK_LIBS
  MLIRIR
  MLIRTensorRTExecutorDialect
)
