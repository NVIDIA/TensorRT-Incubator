add_mlir_tensorrt_library(MLIRTensorRTLinalgSimplifyExtractSlice
  LinalgSimplifyExtractSlice.cpp

  DEPENDS
  MLIRTensorRTPassesIncGenPassIncGen

  MLIR_LIBS PUBLIC
    MLIRIR
    MLIRLinalgDialect
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRTensorTransforms
)
