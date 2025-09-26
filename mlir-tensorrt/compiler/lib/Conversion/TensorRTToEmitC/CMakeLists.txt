add_mlir_tensorrt_library(MLIRTensorRTTensorRTToEmitC
  TensorRTToEmitC.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTCommonIncludes
    MLIRTensorRTTensorRTDialect
    MLIRTensorRTTensorRTUtils
  MLIR_LIBS PUBLIC
    MLIREmitCDialect
    MLIRFuncDialect
    MLIRPass
    MLIRRewrite
    MLIRTransforms
    MLIRTransformUtils
)
