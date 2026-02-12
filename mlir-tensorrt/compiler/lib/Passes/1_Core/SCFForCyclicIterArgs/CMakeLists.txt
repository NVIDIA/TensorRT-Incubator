add_mlir_tensorrt_library(MLIRTensorRTTransformsSCFForCyclicIterArgs
  SCFForCyclicIterArgs.cpp

  DEPENDS
  MLIRTensorRTPassesIncGenPassIncGen

  MLIR_LIBS PUBLIC
    MLIRArithDialect
    MLIRIR
    MLIRPass
    MLIRSCFDialect
)
