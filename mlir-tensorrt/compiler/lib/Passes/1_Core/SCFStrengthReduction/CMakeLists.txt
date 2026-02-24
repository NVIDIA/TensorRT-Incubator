add_mlir_tensorrt_library(MLIRTensorRTTransformsSCFFloatStrengthReduce
  SCFFloatStrengthReduce.cpp

  DEPENDS
  MLIRTensorRTPassesIncGenPassIncGen

  MLIR_LIBS PUBLIC
    MLIRPass
    MLIRArithDialect
    MLIRSCFDialect
    MLIRSCFTransforms
  )
