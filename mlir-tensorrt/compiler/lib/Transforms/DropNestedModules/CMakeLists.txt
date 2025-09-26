add_mlir_tensorrt_library(MLIRTensorRTDropNestedModules
  DropNestedModules.cpp

  DEPENDS
  MLIRTensorRTGenericTransformPassIncGen

  MLIR_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRRewrite
)
