add_mlir_library(MLIRTensorRTAnalysis
  TensorKindAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRBufferizationDialect
  MLIRFunctionInterfaces
  MLIRTensorDialect
  MLIRTensorRTInterfaces
)
