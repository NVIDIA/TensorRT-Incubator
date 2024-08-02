add_mlir_tensorrt_library(MLIRTensorRTStablehloToTensorRT
  StablehloToTensorRT.cpp
  ControlFlowOps.cpp
  ChloToTensorRT.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRRewrite
  MLIRTransforms
  MLIRTransformUtils
  MLIRTensorRTConvertToTensorRTCommon
  MLIRQuantDialect
  MLIRTensorRTDialect
  MLIRTensorRTTensorRTUtils
  MLIRFuncTransforms
  ChloOps
  StablehloOps
)
