add_mlir_tensorrt_library(MLIRTensorRTStablehloToSCF
  StablehloToScf.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRRewrite
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRTensorDialect
  MLIRTensorRTSCFDetensorizeLoops
  MLIRTransformUtils
  StablehloLinalgTransforms
  StablehloOps
  )