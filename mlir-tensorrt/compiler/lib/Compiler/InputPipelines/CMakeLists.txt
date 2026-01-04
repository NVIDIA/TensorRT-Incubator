add_mlir_tensorrt_library(
  MLIRTensorRTStablehloInputPipeline
  StablehloInputPipeline.cpp
  LinalgInputPipeline.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTChloToStablehloExt
    MLIRTensorRTCommandLineExtras
    MLIRTensorRTLinalgElementwiseFusion
    MLIRTensorRTLinalgSimplifyExtractSlice
    MLIRTensorRTTensorExtPadToInsertSlice
    MLIRTensorRTStableHloExtTransforms
    MLIRTensorRTStablehloToPlan
    MLIRTensorRTStablehloToSCF
    MLIRTensorRTTransformsSCFFloatStrengthReduce
    MLIRTensorRTTransformsUnrollForLoops

  MLIR_LIBS PUBLIC
    MLIRTransforms
    MLIRPass
    MLIRIR
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRLinalgTransforms
  )
