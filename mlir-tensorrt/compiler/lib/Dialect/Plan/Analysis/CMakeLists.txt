add_mlir_tensorrt_library(MLIRTensorRTPlanAnalysis
  BoundsAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRTensorRTDialect
  MLIRTensorRTInferTensorValueRangeInterface
  MLIRTensorRTPlanDialect
  MLIRValueBoundsOpInterface
)
