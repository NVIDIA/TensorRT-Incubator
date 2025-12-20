add_mlir_kernel_library(MLIRTensorRTEliminateIndexTypes
  EliminateIndexTypes.cpp

  DEPENDS
  MLIRKernelConversionPassIncGen

  MLIR_LIBS PUBLIC
    MLIRArithDialect
    MLIRFuncDialect
    MLIRFuncTransforms
    MLIRGPUDialect
    MLIRMemRefDialect
    MLIRPass
    MLIRRewrite
    MLIRSCFDialect
    MLIRSCFTransforms
    MLIRTransforms
    MLIRTransformUtils
  )
