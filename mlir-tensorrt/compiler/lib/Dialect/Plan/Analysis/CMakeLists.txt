add_mlir_tensorrt_library(MLIRTensorRTPlanAnalysis
  BoundsAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTPlanDialect
  MLIRTensorRTDialect
  MLIRAnalysis
  MLIRValueBoundsOpInterface
)
