add_mlir_tensorrt_library(MLIRTensorRTSupportStatus
  PARTIAL_SOURCES_INTENDED
  Status.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTCommonIncludes

  DEPENDS
  MLIRTensorRTStatusEnumIncGen
  )

add_mlir_tensorrt_library(MLIRTensorRTCommandLineExtras
  PARTIAL_SOURCES_INTENDED
  CommandLineExtras.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTCommonIncludes
  )
