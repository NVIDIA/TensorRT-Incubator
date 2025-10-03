add_mlir_executor_test_library(MLIRExecutorTestClustering
  TestClustering.cpp

  LINK_LIBS PUBLIC
    MLIRExecutorGenericClustering
  MLIR_LIBS PUBLIC
    MLIRPass
  )

