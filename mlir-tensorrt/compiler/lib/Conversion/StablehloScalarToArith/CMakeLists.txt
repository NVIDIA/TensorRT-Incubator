add_mlir_tensorrt_library(MLIRTensorRTStablehloScalarToArith
  StablehloScalarToArith.cpp

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
  MLIRTensorDialect
  MLIRTransformUtils
  StablehloOps
)