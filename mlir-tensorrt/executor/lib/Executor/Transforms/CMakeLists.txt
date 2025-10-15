add_mlir_executor_library(MLIRTensorRTExecutorTransforms
  AllocsToGlobals.cpp
  BufferizationInterfacesImpl.cpp
  DecomposeAggregateLoadsAndStores.cpp
  EmulateUnsupportedFloats.cpp
  ExpandOps.cpp
  GenerateABIWrappers.cpp
  LowerAggregatesAtFunctionBoundaries.cpp
  LowerGlobals.cpp
  LowerToRuntimeBuiltins.cpp
  PackArguments.cpp
  Passes.cpp
  PopulateFunctionMetadata.cpp

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
    MLIRControlFlowTransforms
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefTransforms
    MLIRPass
    MLIRReconcileUnrealizedCasts
    MLIRSCFToControlFlow
    MLIRSCFTransforms
    MLIRTensorTransforms
  )
