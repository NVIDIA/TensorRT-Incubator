add_mlir_tensorrt_library(MLIRTensorRTMemRefCastElimination
  MemRefCastElimination.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
)