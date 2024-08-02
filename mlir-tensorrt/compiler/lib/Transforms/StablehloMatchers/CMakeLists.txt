add_mlir_tensorrt_library(MLIRTensorRTStablehloMatchers
  StablehloMatchers.cpp

  DEPENDS
  MLIRTensorRTGenericTransformPassIncGen
  
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRRewrite
  MLIRPass
  StablehloOps
  MLIRTensorDialect
)
