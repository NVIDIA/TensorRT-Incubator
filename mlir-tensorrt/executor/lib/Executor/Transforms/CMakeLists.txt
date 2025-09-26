add_mlir_executor_library(MLIRTensorRTExecutorTransforms
  AllocsToGlobals.cpp
  DecomposeAggregateLoadsAndStores.cpp
  ExpandOps.cpp
  LowerAggregatesAtFunctionBoundaries.cpp
  LowerGlobals.cpp
  LowerToRuntimeBuiltins.cpp
  PackArguments.cpp
  Passes.cpp
  PopulateFunctionMetadata.cpp
  EmulateUnsupportedFloats.cpp

  DEPENDS
  MLIRTensorRTExecutorTransformsPassIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTConvertToExecutorCommon
    MLIRTensorRTExecutorDialect
    MLIRTensorRTExecutorUtils
    MLIRTensorRTRuntimeExecutableInterface
    MLIRTensorRTStdToExecutor
  MLIR_LIBS PUBLIC
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
    MLIRTensorTransforms
  )
