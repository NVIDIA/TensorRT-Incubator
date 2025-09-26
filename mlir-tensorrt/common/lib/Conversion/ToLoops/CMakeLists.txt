add_mlir_tensorrt_library(MLIRTensorRTCommonConvertToLoops
  ConvertToLoops.cpp

  DEPENDS
  MLIRTensorRTCommonConversionPassesIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTCommonIncludes
    MLIRTensorRTCommonLinalgExtTransforms
    MLIRTensorRTCommonToLoopsOpInterface
  MLIR_LIBS PUBLIC
    MLIRIR
    MLIRLinalgDialect
    MLIRSCFDialect
    MLIRTensorDialect
  )