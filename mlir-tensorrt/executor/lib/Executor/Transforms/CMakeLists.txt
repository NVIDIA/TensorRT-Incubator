add_mlir_executor_library(MLIRTensorRTExecutorTransforms
  AllocsToGlobals.cpp
  DecomposeAggregateLoadsAndStores.cpp
  ExpandOps.cpp
  LowerGlobals.cpp
  LowerToRuntimeBuiltins.cpp
  Passes.cpp
  PackArguments.cpp
  PopulateFunctionMetadata.cpp
  EmulateUnsupportedFloats.cpp

  DEPENDS
  MLIRTensorRTExecutorTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineToStandard
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRComplexToStandard
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefTransforms
  MLIRPass
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRSCFTransforms
  MLIRTensorRTConvertToExecutorCommon
  MLIRTensorRTExecutorDialect
  MLIRTensorRTExecutorUtils
  MLIRTensorRTStdToExecutor
  MLIRTensorTransforms
  )
