add_mlir_tensorrt_library(MLIRTensorRTStablehloToTensorRT
  StablehloToTensorRT.cpp
  ControlFlowOps.cpp
  ChloToTensorRT.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen
  MLIRTensorRTEnumConverterGen

  LINK_LIBS PUBLIC
  ChloOps
  MLIRArithDialect
  MLIRFuncTransforms
  MLIRQuantDialect
  MLIRRewrite
  MLIRTensorRTConvertToTensorRTCommon
  MLIRTensorRTDialect
  MLIRTensorRTStableHloExtUtils
  MLIRTensorRTStablehloMatchers
  MLIRTensorRTTensorRTUtils
  MLIRTransforms
  MLIRTransformUtils
  StablehloOps
)
