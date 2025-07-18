add_mlir_tensorrt_library(MLIRTensorRTTransformsUnrollForLoops
  UnrollForLoops.cpp

  DEPENDS
  MLIRTensorRTGenericTransformPassIncGen


  LINK_LIBS PUBLIC
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSCFUtils
  )
