add_mlir_tensorrt_library(MLIRTensorRTPlanAnalysis
  BoundsAnalysis.cpp
  DimAnalysis.cpp
  DimAnalysisConstraints.cpp
  DimAnalysisTypes.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTTensorRTDialect
    MLIRTensorRTInferTensorValueRangeInterface
    MLIRTensorRTPlanDialect
    StablehloOps
  MLIR_LIBS PUBLIC
    MLIRAnalysis
    MLIRValueBoundsOpInterface
)
