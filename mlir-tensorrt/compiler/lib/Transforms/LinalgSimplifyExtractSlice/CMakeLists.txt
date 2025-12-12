add_mlir_tensorrt_library(MLIRTensorRTLinalgSimplifyExtractSlice
  LinalgSimplifyExtractSlice.cpp

  DEPENDS
  MLIRTensorRTGenericTransformPassIncGen

  MLIR_LIBS PUBLIC
    MLIRIR
    MLIRLinalgDialect
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRTensorTransforms
)
