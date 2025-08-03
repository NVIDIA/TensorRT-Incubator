add_mlir_tensorrt_library(MLIRTensorRTStablehloToTensorRT
  ControlFlowOps.cpp
  ChloToTensorRT.cpp
  ReductionConversions.cpp
  StablehloToTensorRT.cpp

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
  MLIRTensorRTTensorRTDialect
  MLIRTensorRTStableHloExtUtils
  MLIRTensorRTStablehloMatchers
  MLIRTensorRTTensorRTUtils
  MLIRTensorRTUtilsShapeInfo
  MLIRTransforms
  MLIRTransformUtils
  StablehloOps
)
