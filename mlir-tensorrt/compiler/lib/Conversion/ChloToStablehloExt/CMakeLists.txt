add_mlir_tensorrt_library(MLIRTensorRTChloToStablehloExt
  ChloToStablehloExt.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
  ChloOps
  MLIRDialectUtils
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRTensorRTDialect
  MLIRTransformUtils
  StablehloOps
  StablehloPasses
  )