add_mlir_tensorrt_library(MLIRTensorRTTensorRTToEmitC
  TensorRTToEmitC.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRRewrite
  MLIRTransforms
  MLIRTransformUtils  
  MLIRFuncDialect
  MLIREmitCDialect
  MLIRPass

  MLIRTensorRTDialect
)
