add_mlir_tensorrt_test_library(MLIRTensorRTTestDimAnalysis
  TestDimAnalysis.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTPlanAnalysis
    StablehloOps

  MLIR_LIBS PUBLIC
    MLIRPass
)
