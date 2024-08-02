add_mlir_tensorrt_library(MLIRTensorRTSCFDetensorizeLoops
  SCFDetensorizeLoops.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTensorRTAnalysis
)
