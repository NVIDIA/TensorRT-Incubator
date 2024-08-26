add_mlir_executor_library(MLIRTensorRTExecutorTransforms
  AllocsToGlobals.cpp
  DecomposeAggregateLoadsAndStores.cpp
  ExpandOps.cpp
  LowerGlobals.cpp
  LowerToRuntimeBuiltins.cpp
  Passes.cpp
  PackArguments.cpp
  PopulateFunctionMetadata.cpp

  DEPENDS
  MLIRTensorRTExecutorTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineToStandard
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRComplexToStandard
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRReconcileUnrealizedCasts
  MLIRTensorRTConvertToExecutorCommon
  MLIRTensorRTExecutorDialect
  MLIRTensorRTExecutorUtils
  MLIRTensorRTStdToExecutor
)
