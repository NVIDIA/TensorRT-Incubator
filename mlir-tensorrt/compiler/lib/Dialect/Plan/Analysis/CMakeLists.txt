add_mlir_tensorrt_library(MLIRTensorRTPlanAnalysis
  BoundsAnalysis.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTTensorRTDialect
    MLIRTensorRTInferTensorValueRangeInterface
    MLIRTensorRTPlanDialect
  MLIR_LIBS PUBLIC
    MLIRAnalysis
    MLIRValueBoundsOpInterface
)
