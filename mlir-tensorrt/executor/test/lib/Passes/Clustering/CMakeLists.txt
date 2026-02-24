add_mlir_executor_test_library(MLIRExecutorTestClustering
  TestClustering.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTExecutorPasses
  MLIR_LIBS PUBLIC
    MLIRPass
  )
