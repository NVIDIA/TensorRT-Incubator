add_mlir_executor_unittest(MLIRExecutorADTExtrasTests
  ADTExtrasTests.cpp
  PARTIAL_SOURCES_INTENDED
  )

target_link_libraries(MLIRExecutorADTExtrasTests
  PRIVATE
  MLIRTensorRTCommonIncludes
  )

add_mlir_executor_unittest(MLIRExecutorStatusTests
  StatusTests.cpp
  PARTIAL_SOURCES_INTENDED
  )
target_link_libraries(MLIRExecutorStatusTests
  PRIVATE
  MLIRTensorRTSupportStatus
  )

add_mlir_executor_unittest(MLIRExecutorSlotAssignmentTests
  SlotAssignmentTest.cpp
  PARTIAL_SOURCES_INTENDED
  )
target_link_libraries(MLIRExecutorSlotAssignmentTests
  PRIVATE
  MLIRTensorRTSlotAssignment
  )
