add_mlir_tensorrt_dialect_library(MLIRTensorRTPlanDialect
  PlanInterfaces.cpp
  PlanOps.cpp

  DEPENDS
  MLIRPlanOpsIncGen
  MLIRTensorRTPlanDialectEnumsIncGen
  MLIRTensorRTPlanDialectAttributesIncGen
  MLIRTensorRTPlanDialectAttrInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRControlFlowInterfaces
  MLIRDialect
  MLIRExecutorGenericClustering
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRSupport
  MLIRTensorRTInterfaces
  MLIRTensorRTSupportStatus
  MLIRTensorRTUtilsShapeInfo
  )
