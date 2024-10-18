add_mlir_tensorrt_dialect_library(MLIRTensorRTPlanDialect
  PlanInterfaces.cpp
  PlanOps.cpp
  BuiltinClusterKinds.cpp

  DEPENDS
  MLIRPlanOpsIncGen
  MLIRTensorRTPlanDialectEnumsIncGen
  MLIRTensorRTPlanDialectAttributesIncGen
  MLIRTensorRTPlanDialectAttrInterfacesIncGen

  LINK_LIBS PUBLIC
  ChloOps
  MLIRControlFlowInterfaces
  MLIRDialect
  MLIRExecutorGenericClustering
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRSupport
  MLIRTensorRTConvertToTensorRTCommon
  MLIRTensorRTInterfaces
  MLIRTensorRTStablehloScalarToArith
  MLIRTensorRTStablehloToTensorRT
  MLIRTensorRTSupportStatus
  StablehloOps
)
