add_mlir_tensorrt_library(MLIRTensorRTStablehloToPlan
  StablehloToPlan.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTPlanDialect
    StablehloOps
    MLIRTensorRTExecutorDialect
    MLIRTensorRTExecutorUtils
  MLIR_LIBS PUBLIC
    MLIRPass
    MLIRRewrite
    MLIRTransformUtils
)
