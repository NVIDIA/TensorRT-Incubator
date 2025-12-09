add_mlir_tensorrt_library(MLIRTensorRTLinalgElementwiseFusion
  LinalgElementwiseFusion.cpp

  DEPENDS
  MLIRTensorRTGenericTransformPassIncGen

  MLIR_LIBS PUBLIC
    MLIRIR
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRLinalgTransforms
)
