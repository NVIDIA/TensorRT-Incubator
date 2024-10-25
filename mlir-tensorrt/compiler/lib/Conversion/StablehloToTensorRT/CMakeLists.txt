add_mlir_tensorrt_library(MLIRTensorRTStablehloToTensorRT
  StablehloToTensorRT.cpp
  ControlFlowOps.cpp
  ChloToTensorRT.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen
  MLIRTensorRTEnumConverterGen

  LINK_LIBS PUBLIC
  ChloOps
  MLIRFuncTransforms
  MLIRQuantDialect
  MLIRRewrite
  MLIRTensorRTConvertToTensorRTCommon
  MLIRTensorRTDialect
  MLIRTensorRTStablehloInputPreprocessing
  MLIRTensorRTStablehloMatchers
  MLIRTensorRTTensorRTUtils
  MLIRTransforms
  MLIRTransformUtils
  StablehloOps
)
