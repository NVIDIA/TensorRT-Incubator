add_mlir_tensorrt_library(MLIRTensorRTDropNestedModules
  DropNestedModules.cpp

  DEPENDS
  MLIRTensorRTPassesIncGenPassIncGen

  MLIR_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRRewrite
)
