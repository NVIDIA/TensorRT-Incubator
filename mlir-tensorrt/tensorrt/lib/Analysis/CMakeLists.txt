add_mtrtd_library(MLIRTensorRTAnalysis
  TensorKindAnalysis.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTCommonTensorKindInterface
    MLIRTensorRTDialectIncludes
  MLIR_LIBS PUBLIC
    MLIRAnalysis
    MLIRBufferizationDialect
    MLIRFunctionInterfaces
    MLIRTensorDialect
)
