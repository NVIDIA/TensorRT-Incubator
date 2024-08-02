add_mlir_tensorrt_library(MLIRTensorRTDropNestedModules
  DropNestedModules.cpp

  DEPENDS
  MLIRTensorRTGenericTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRRewrite
)
