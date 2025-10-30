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
    MLIRExecutorGenericClustering
    MLIRTensorRTCommonBoundsAttrInterface
    MLIRTensorRTCommonTensorKindInterface
    MLIRTensorRTExecutorDialect
    MLIRTensorRTInferTensorValueRangeInterface
    MLIRTensorRTSupportStatus
    MLIRTensorRTUtilsShapeInfo
  MLIR_LIBS PUBLIC
    MLIRBufferizationDialect
    MLIRControlFlowInterfaces
    MLIRDialect
    MLIRFuncDialect
    MLIRInferTypeOpInterface
    MLIRIR
    MLIRSupport
    MLIRTensorDialect
  )
