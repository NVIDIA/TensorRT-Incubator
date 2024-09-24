add_mlir_tensorrt_library(MLIRTensorRTSCFDetensorizeLoops
  SCFDetensorizeLoops.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTensorRTAnalysis
  MLIRTransformUtils
)
