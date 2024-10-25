add_mlir_executor_unittest(MLIRExecutorADTExtrasTests
  ADTExtrasTests.cpp
  PARTIAL_SOURCES_INTENDED
  )

add_mlir_executor_unittest(MLIRExecutorStatusTests
  StatusTests.cpp
  PARTIAL_SOURCES_INTENDED
  )
target_link_libraries(MLIRExecutorStatusTests
  PRIVATE
  MLIRTensorRTSupportStatus
  )
