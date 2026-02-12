add_mlir_tensorrt_library(MLIRTensorRTLinalgElementwiseFusion
  LinalgElementwiseFusion.cpp

  DEPENDS
  MLIRTensorRTPassesIncGenPassIncGen

  MLIR_LIBS PUBLIC
    MLIRIR
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRLinalgTransforms
)
