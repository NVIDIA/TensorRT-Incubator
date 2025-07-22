add_mlir_tensorrt_library(MLIRTensorRTSupportStatus
  PARTIAL_SOURCES_INTENDED
  Status.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTCommonIncludes

  DEPENDS
  MLIRTensorRTStatusEnumIncGen
)

