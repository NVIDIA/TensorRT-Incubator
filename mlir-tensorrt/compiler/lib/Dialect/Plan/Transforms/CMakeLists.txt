add_mlir_tensorrt_library(MLIRTensorRTPlanTransforms
  AllocTensors.cpp
  CreateClosedRegions.cpp
  CreateShapeFuncs.cpp
  Bufferize.cpp
  EliminateShapeOps.cpp
  MaterializeShapeCalculations.cpp
  StablehloClustering.cpp
  OutlineClusters.cpp
  Passes.cpp
  PostClusteringValidation.cpp
  RefineTypes.cpp
  PopulateFunctionBoundsAttributes.cpp

  DEPENDS
  MLIRTensorRTPlanTransformsPassIncGen

  LINK_LIBS PUBLIC

  MLIRBufferizationPipelines
  MLIRExecutorGenericClustering
  MLIRFuncTransforms
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTensorRTAnalysis
  MLIRTensorRTCUDADialect
  MLIRTensorRTDialect
  MLIRTensorRTDuplicateFunctionElimination
  MLIRTensorRTExecutorDialect
  MLIRTensorRTMemRefCastElimination
  MLIRTensorRTPlanAnalysis
  MLIRTensorRTPlanDialect
  MLIRTensorRTStableHloExtTransforms
  MLIRTensorRTStablehloScalarToArith
  MLIRTensorRTStablehloToTensorRT
  MLIRTensorRTTensorRTRuntimeDialect
  MLIRTransforms
  StablehloOps
)
