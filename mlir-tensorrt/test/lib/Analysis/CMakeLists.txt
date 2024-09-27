add_mlir_tensorrt_test_library(MLIRTensorRTTestTensorKindAnalysis
  TestTensorKindAnalysis.cpp
  TestBoundsAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTensorRTAnalysis
  MLIRTensorRTPlanAnalysis
  )
