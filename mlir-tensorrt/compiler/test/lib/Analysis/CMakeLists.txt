add_mlir_tensorrt_test_library(MLIRTensorRTCompilerAnalysisTestPasses
  TestBoundsAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTPlanAnalysis

  MLIR_LIBS PUBLIC
  MLIRPass
  )
