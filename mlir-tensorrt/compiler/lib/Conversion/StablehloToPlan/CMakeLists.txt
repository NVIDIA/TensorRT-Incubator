add_mlir_tensorrt_library(MLIRTensorRTStablehloToPlan
  StablehloToPlan.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRRewrite
  MLIRTensorRTPlanDialect
  MLIRTransformUtils
  StablehloOps
)
