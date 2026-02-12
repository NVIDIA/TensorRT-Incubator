add_mlir_tensorrt_library(MLIRTensorRTMemRefCastElimination
  MemRefCastElimination.cpp

  DEPENDS
  MLIRTensorRTPassesIncGenPassIncGen

  MLIR_LIBS PUBLIC
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
    MLIRSCFDialect
    MLIRTransformUtils
)
