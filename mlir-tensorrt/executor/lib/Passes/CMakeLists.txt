include_directories(${CMAKE_CURRENT_BINARY_DIR})
get_target_property(common_include_dirs MLIRTensorRTCommonIncludes INTERFACE_INCLUDE_DIRECTORIES)
include_directories(${common_include_dirs})

add_mlir_pdll_library(MLIRExecutorStdToExectutorPatternsIncGen
  StdToExecutor.pdll
  StdToExecutor.pdll.h.inc
  )

add_mlir_executor_library(MLIRTensorRTExecutorPasses
  # Conversion sources
  ConvertToExecutorCommon.cpp
  ExecutorToExecutor.cpp
  MemRefToExecutor.cpp
  StdToExecutor.cpp
  LinalgToExecutor.cpp
  TVMFFIUtils.cpp
  # Transform sources
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
  # Clustering sources
  Clustering/Clustering.cpp
  Clustering/Patterns.cpp

  DEPENDS
  MLIRTensorRTExecutorPassesIncGen
  MLIRExecutorStdToExectutorPatternsIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTExecutorDialect
    MLIRTensorRTExecutorUtils
    MLIRExecutorIncludes
    MLIRTensorRTRuntimeExecutableInterface
    MLIRTensorRTSupportArtifactManager
    MLIRTensorRTModuleLikePass
    DLPack::Headers
    tvm_ffi_header

  MLIR_LIBS PUBLIC
    MLIRAffineToStandard
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
    MLIRComplexToStandard
    MLIRControlFlowDialect
    MLIRControlFlowTransforms
    MLIRFuncDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRMathTransforms
    MLIRMemRefDialect
    MLIRMemRefTransforms
    MLIRParser
    MLIRPass
    MLIRReconcileUnrealizedCasts
    MLIRRewrite
    MLIRSCFDialect
    MLIRSCFToControlFlow
    MLIRSCFTransforms
    MLIRTensorTransforms
    MLIRTransforms
    MLIRTransformUtils
    MLIRDLTIDialect
  )

# Aliases for external projects that still reference the old split targets
add_library(MLIRTensorRTConvertToExecutorCommon ALIAS MLIRTensorRTExecutorPasses)
add_library(MLIRTensorRTStdToExecutor ALIAS MLIRTensorRTExecutorPasses)
add_library(MLIRTensorRTExecutorTransforms ALIAS MLIRTensorRTExecutorPasses)
add_library(MLIRExecutorGenericClustering ALIAS MLIRTensorRTExecutorPasses)
