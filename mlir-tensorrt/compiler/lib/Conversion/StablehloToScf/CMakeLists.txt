add_mlir_tensorrt_library(MLIRTensorRTStablehloToSCF
  StablehloToScf.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTSCFDetensorizeLoops
    StablehloLinalgTransforms
    StablehloOps
  MLIR_LIBS PUBLIC
    MLIRArithDialect
    MLIRDialectUtils
    MLIRIR
    MLIRMathDialect
    MLIRPass
    MLIRRewrite
    MLIRSCFDialect
    MLIRSCFTransforms
    MLIRTensorDialect
    MLIRTransformUtils
  )