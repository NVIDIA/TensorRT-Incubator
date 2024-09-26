add_mlir_tensorrt_library(MLIRTensorRTTensorRTToEmitC
  TensorRTToEmitC.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIREmitCDialect
  MLIRFuncDialect
  MLIRPass
  MLIRRewrite
  MLIRTensorRTDialect
  MLIRTensorRTTensorRTUtils
  MLIRTransforms
  MLIRTransformUtils
)
