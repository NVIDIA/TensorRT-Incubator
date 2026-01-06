add_mlir_executor_library(MLIRTensorRTExecutorTransforms
  AllocsToGlobals.cpp
  BufferizationInterfacesImpl.cpp
  DecomposeAggregateLoadsAndStores.cpp
  EmulateUnsupportedFloats.cpp
  ExpandMathOps.cpp
  ExpandOps.cpp
  GenerateABIWrappers.cpp
  LowerABIOps.cpp
  LowerAggregatesAtFunctionBoundaries.cpp
  LowerGlobals.cpp
  LowerToRuntimeBuiltins.cpp
  PackArguments.cpp
  Passes.cpp
  PopulateFunctionMetadata.cpp
  SerializeArtifacts.cpp

  DEPENDS
  MLIRTensorRTExecutorTransformsPassIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTConvertToExecutorCommon
    MLIRTensorRTExecutorDialect
    MLIRTensorRTExecutorUtils
    MLIRTensorRTRuntimeExecutableInterface
    MLIRTensorRTStdToExecutor
    MLIRTensorRTSupportArtifactManager
  MLIR_LIBS PUBLIC
    MLIRAffineToStandard
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
    MLIRComplexToStandard
    MLIRControlFlowTransforms
    MLIRFuncDialect
    MLIRIR
    MLIRMathTransforms
    MLIRMemRefTransforms
    MLIRPass
    MLIRReconcileUnrealizedCasts
    MLIRSCFToControlFlow
    MLIRSCFTransforms
    MLIRTensorTransforms
  )
