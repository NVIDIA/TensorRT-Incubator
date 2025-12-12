add_mlir_kernel_library(MLIRTensorRTComplexToStdExt
  ComplexToStdExt.cpp

  DEPENDS
  MLIRKernelConversionPassIncGen

  MLIR_LIBS PUBLIC
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRComplexDialect
    MLIRComplexToStandard
    MLIRFuncDialect
    MLIRFuncTransforms
    MLIRGPUDialect
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRRewrite
    MLIRSCFDialect
    MLIRSCFTransforms
    MLIRTensorDialect
    MLIRTransforms
    MLIRTransformUtils
  LINK_LIBS PUBLIC
    MLIRTensorRTExecutorDialect
    MLIRExecutorIncludes
  )
