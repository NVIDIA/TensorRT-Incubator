add_mlir_tensorrt_library(MLIRTensorRTStablehloRaiseQDQ
  StablehloRaiseQDQ.cpp

  DEPENDS
  MLIRTensorRTGenericTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRRewrite
  StablehloOps
  MLIRFuncDialect
)