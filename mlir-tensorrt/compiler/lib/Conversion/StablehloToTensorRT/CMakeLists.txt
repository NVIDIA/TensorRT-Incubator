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
    MLIRTensorRTConvertToTensorRTCommon
    MLIRTensorRTTensorRTDialect
    MLIRTensorRTStableHloExtUtils
    MLIRTensorRTStablehloMatchers
    MLIRTensorRTTensorRTUtils
    MLIRTensorRTUtilsShapeInfo
    StablehloOps
  MLIR_LIBS PUBLIC
    MLIRArithDialect
    MLIRFuncTransforms
    MLIRQuantDialect
    MLIRRewrite
    MLIRTransforms
    MLIRTransformUtils
)
