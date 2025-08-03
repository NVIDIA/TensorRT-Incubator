add_mlir_tensorrt_dialect_library(MLIRTensorRTPlanDialect
  PlanAttributes.cpp
  PlanInterfaces.cpp
  PlanOps.cpp
  PlanTypes.cpp

  DEPENDS
  MLIRPlanOpsIncGen
  MLIRTensorRTPlanDialectEnumsIncGen
  MLIRTensorRTPlanDialectAttributesIncGen
  MLIRTensorRTPlanDialectAttrInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRControlFlowInterfaces
  MLIRDialect
  MLIRExecutorGenericClustering
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRSupport
  MLIRTensorDialect
  MLIRTensorRTCommonBoundsAttrInterface
  MLIRTensorRTInferTensorValueRangeInterface
  MLIRTensorRTCommonTensorKindInterface
  MLIRTensorRTSupportStatus
  MLIRTensorRTUtilsShapeInfo
  )
