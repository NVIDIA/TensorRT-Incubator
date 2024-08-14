add_mlir_tensorrt_library(MLIRTensorRTStablehloToTensorRT
  StablehloToTensorRT.cpp
  ControlFlowOps.cpp
  ChloToTensorRT.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
  ChloOps
  MLIRFuncTransforms
  MLIRQuantDialect
  MLIRRewrite
  MLIRTensorRTConvertToTensorRTCommon
  MLIRTensorRTDialect
  MLIRTensorRTStableHloExtUtils
  MLIRTensorRTStablehloInputPreprocessing
  MLIRTensorRTTensorRTUtils
  MLIRTransforms
  MLIRTransformUtils
  StablehloOps
)
