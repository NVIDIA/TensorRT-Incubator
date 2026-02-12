add_mlir_tensorrt_library(MLIRTensorRTTransformsUnrollForLoops
  UnrollForLoops.cpp

  DEPENDS
  MLIRTensorRTPassesIncGenPassIncGen


  MLIR_LIBS PUBLIC
    MLIRPass
    MLIRSCFDialect
    MLIRSCFTransforms
    MLIRSCFUtils
  )
