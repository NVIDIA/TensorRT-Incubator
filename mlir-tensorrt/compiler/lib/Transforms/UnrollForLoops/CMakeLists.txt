add_mlir_tensorrt_library(MLIRTensorRTTransformsUnrollForLoops
  UnrollForLoops.cpp

  DEPENDS
  MLIRTensorRTGenericTransformPassIncGen


  MLIR_LIBS PUBLIC
    MLIRPass
    MLIRSCFDialect
    MLIRSCFTransforms
    MLIRSCFUtils
  )
