add_mlir_tensorrt_library(MLIRTensorRTStablehloToSCF
  StablehloToScf.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransformUtils
  StablehloOps
  )