add_mlir_tensorrt_dialect_library(MLIRTensorRTPlanDialect
  PlanOps.cpp
  BuiltinClusterKinds.cpp

  DEPENDS
  MLIRPlanOpsIncGen
  MLIRTensorRTPlanDialectEnumsIncGen
  MLIRTensorRTPlanDialectAttributesIncGen
  MLIRTensorRTPlanDialectAttrInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDialect
  MLIRSupport
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRControlFlowInterfaces
  MLIRTensorRTInterfaces
  MLIRTensorRTGenericClustering
  MLIRTensorRTConvertToTensorRTCommon
  MLIRTensorRTStablehloScalarToArith
  MLIRTensorRTStablehloToTensorRT
  StablehloOps
  ChloOps
)
