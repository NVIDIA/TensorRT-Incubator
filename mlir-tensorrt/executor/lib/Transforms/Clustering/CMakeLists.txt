add_mlir_executor_library(MLIRExecutorGenericClustering
  Clustering.cpp
  Patterns.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRTransformUtils
)