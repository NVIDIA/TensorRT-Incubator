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
  MLIRBufferizationDialect
  MLIRDialect
  MLIRExecutorGenericClustering
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRSupport
  MLIRTensorDialect
  MLIRTensorRTInferTensorValueRangeInterface
  MLIRTensorRTInterfaces
  MLIRTensorRTSupportStatus
  MLIRTensorRTUtilsShapeInfo
  )
