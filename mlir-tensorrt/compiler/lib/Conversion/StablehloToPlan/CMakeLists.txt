add_mlir_tensorrt_library(MLIRTensorRTStablehloToPlan
  StablehloToPlan.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTPlanDialect
    StablehloOps
    MLIRTensorRTExecutorDialect
    MLIRTensorRTExecutorUtils
    MLIRTensorRTCUDAUtils
  MLIR_LIBS PUBLIC
    MLIRControlFlowDialect
    MLIRPass
    MLIRRewrite
    MLIRTransformUtils
)
