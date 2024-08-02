add_mlir_tensorrt_library(MLIRTensorRTStablehloInputPreprocessing
  StablehloPrepareConvolution.cpp
  StablehloInputPreprocessing.cpp
  StablehloPrepareScatter.cpp

  DEPENDS
  MLIRTensorRTGenericTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRRewrite
  StablehloOps
  StablehloPasses
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRSCFTransforms
  MLIRTensorRTStableHloExtUtils
)
