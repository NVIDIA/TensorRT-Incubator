add_library(MLIRTensorRTSupportStatus INTERFACE)
target_link_libraries(MLIRTensorRTSupportStatus INTERFACE
  MLIRTensorRTCommonIncludes
)
add_dependencies(MLIRTensorRTSupportStatus MLIRTensorRTStatusEnumIncGen)

add_mlir_tensorrt_library(MLIRTensorRTCommandLineExtras
  PARTIAL_SOURCES_INTENDED
  CommandLineExtras.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTCommonIncludes
  )
