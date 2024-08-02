add_mlir_tensorrt_library(MLIRTensorRTStablehloScalarToArith
  StablehloScalarToArith.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRPass
  MLIRRewrite
  StablehloOps
  MLIRArithDialect
  MLIRSCFDialect
  MLIRMathDialect
  MLIRTensorDialect
)