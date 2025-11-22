add_mlir_tensorrt_library(MLIRTensorRTSCFDetensorizeLoops
  SCFDetensorizeLoops.cpp

  DEPENDS
  MLIRTensorRTGenericTransformPassIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTAnalysis
  MLIR_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRTransformUtils
)
