add_mlir_tensorrt_test_library(MLIRTensorRTTestTensorKindAnalysis
  TestTensorKindAnalysis.cpp
  TestBoundsAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTAnalysis
  MLIRTensorRTPlanAnalysis

  MLIR_LIBS PUBLIC
  MLIRPass
  )
