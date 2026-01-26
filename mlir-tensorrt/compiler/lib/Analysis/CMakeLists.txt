add_mlir_tensorrt_library(MLIRTensorRTCompilerAnalysis
  AliasAnalysis.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTExecutorDialect
  MLIR_LIBS PUBLIC
    MLIRAnalysis
    MLIRFunctionInterfaces
)

