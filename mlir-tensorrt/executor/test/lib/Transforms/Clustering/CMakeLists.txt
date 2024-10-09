add_mlir_executor_test_library(MLIRExecutorTestClustering
  TestClustering.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRExecutorGenericClustering
  MLIRPass
  )

