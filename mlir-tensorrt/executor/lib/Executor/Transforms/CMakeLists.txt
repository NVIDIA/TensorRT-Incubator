add_mlir_library(MLIRTensorRTExecutorTransforms
  AllocsToGlobals.cpp
  ExpandOps.cpp
  LowerGlobals.cpp
  Passes.cpp
  PackArguments.cpp
  PopulateFunctionMetadata.cpp

  DEPENDS
  MLIRTensorRTExecutorTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRTensorRTExecutorDialect
  MLIRTensorRTExecutorUtils
  MLIRLinalgTransforms
  MLIRTensorRTConvertToExecutorCommon
  MLIRTensorRTStdToExecutor
  MLIRAffineToStandard
)
