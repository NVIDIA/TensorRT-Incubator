add_mlir_tensorrt_test_library(MLIRTensorRTTestCUDA
  TestCUDASideEffects.cpp

  MLIR_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSideEffectInterfaces
)
