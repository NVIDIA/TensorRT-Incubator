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

  StablehloOps
  MLIRBufferizationPipelines
  MLIRIR
  MLIRPass
  MLIRTensorRTPlanDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransforms
  MLIRFuncTransforms

  MLIRTensorRTAnalysis
  MLIRTensorRTGenericClustering
  MLIRTensorRTDialect
  MLIRTensorRTStablehloToTensorRT
  MLIRTensorRTStablehloScalarToArith
  MLIRTensorRTPlanAnalysis
)
