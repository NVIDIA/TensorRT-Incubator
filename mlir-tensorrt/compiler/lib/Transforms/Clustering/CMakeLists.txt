add_mlir_tensorrt_library(MLIRTensorRTGenericClustering
  Clustering.cpp
  Patterns.cpp

  DEPENDS
  MLIRTensorRTGenericTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSCFDialect
)